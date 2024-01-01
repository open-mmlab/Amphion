# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import torch
import json
import itertools
import accelerate
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from librosa.filters import mel as librosa_mel_fn

from accelerate.logging import get_logger
from pathlib import Path

from utils.io import save_audio
from utils.data_utils import *
from utils.util import (
    Logger,
    ValueWindow,
    remove_older_ckpt,
    set_all_random_seed,
    save_config,
)
from utils.mel import extract_mel_features
from models.vocoders.vocoder_trainer import VocoderTrainer
from models.vocoders.gan.gan_vocoder_dataset import (
    GANVocoderDataset,
    GANVocoderCollator,
)

from models.vocoders.gan.generator.bigvgan import BigVGAN
from models.vocoders.gan.generator.hifigan import HiFiGAN
from models.vocoders.gan.generator.melgan import MelGAN
from models.vocoders.gan.generator.nsfhifigan import NSFHiFiGAN
from models.vocoders.gan.generator.apnet import APNet

from models.vocoders.gan.discriminator.mpd import MultiPeriodDiscriminator
from models.vocoders.gan.discriminator.mrd import MultiResolutionDiscriminator
from models.vocoders.gan.discriminator.mssbcqtd import MultiScaleSubbandCQTDiscriminator
from models.vocoders.gan.discriminator.msd import MultiScaleDiscriminator
from models.vocoders.gan.discriminator.msstftd import MultiScaleSTFTDiscriminator

from models.vocoders.gan.gan_vocoder_inference import vocoder_inference

supported_generators = {
    "bigvgan": BigVGAN,
    "hifigan": HiFiGAN,
    "melgan": MelGAN,
    "nsfhifigan": NSFHiFiGAN,
    "apnet": APNet,
}

supported_discriminators = {
    "mpd": MultiPeriodDiscriminator,
    "msd": MultiScaleDiscriminator,
    "mrd": MultiResolutionDiscriminator,
    "msstftd": MultiScaleSTFTDiscriminator,
    "mssbcqtd": MultiScaleSubbandCQTDiscriminator,
}


class GANVocoderTrainer(VocoderTrainer):
    def __init__(self, args, cfg):
        super().__init__()

        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        # Init accelerator
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            self.logger = get_logger(args.exp_name, log_level=args.log_level)

        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New training process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")
        self.logger.debug(f"Using {args.log_level.upper()} logging level.")
        self.logger.info(f"Experiment name: {args.exp_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # Init training status
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0

        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        self.logger.info(
            "Max epoch: {}".format(
                self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
            )
        )

        # Check potential erorrs
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.run_eval = self.cfg.train.run_eval

        # Set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # Build dataloader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # Build model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.generator, self.discriminators = self._build_model()
            end = time.monotonic_ns()
            self.logger.debug(self.generator)
            for _, discriminator in self.discriminators.items():
                self.logger.debug(discriminator)
            self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
            self.logger.info(f"Model parameters: {self._count_parameters()/1e6:.2f}M")

        # Build optimizers and schedulers
        with self.accelerator.main_process_first():
            self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            (
                self.generator_optimizer,
                self.discriminator_optimizer,
            ) = self._build_optimizer()
            (
                self.generator_scheduler,
                self.discriminator_scheduler,
            ) = self._build_scheduler()
            end = time.monotonic_ns()
            self.logger.info(
                f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
            )

        # Accelerator preparing
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        (
            self.train_dataloader,
            self.valid_dataloader,
            self.generator,
            self.generator_optimizer,
            self.discriminator_optimizer,
            self.generator_scheduler,
            self.discriminator_scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.valid_dataloader,
            self.generator,
            self.generator_optimizer,
            self.discriminator_optimizer,
            self.generator_scheduler,
            self.discriminator_scheduler,
        )
        for key, discriminator in self.discriminators.items():
            self.discriminators[key] = self.accelerator.prepare_model(discriminator)
        end = time.monotonic_ns()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # Build criterions
        with self.accelerator.main_process_first():
            self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterions = self._build_criterion()
            end = time.monotonic_ns()
            self.logger.info(f"Building criterion done in {(end - start) / 1e6:.2f}ms")

        # Resume checkpoints
        with self.accelerator.main_process_first():
            if args.resume_type:
                self.logger.info("Resuming from checkpoint...")
                start = time.monotonic_ns()
                ckpt_path = Path(args.checkpoint)
                if self._is_valid_pattern(ckpt_path.parts[-1]):
                    ckpt_path = self._load_model(
                        None, args.checkpoint, args.resume_type
                    )
                else:
                    ckpt_path = self._load_model(
                        args.checkpoint, resume_type=args.resume_type
                    )
                end = time.monotonic_ns()
                self.logger.info(
                    f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                )
                self.checkpoints_path = json.load(
                    open(os.path.join(ckpt_path, "ckpts.json"), "r")
                )

            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # Save config
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

    def _build_dataset(self):
        return GANVocoderDataset, GANVocoderCollator

    def _build_criterion(self):
        class feature_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(feature_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, fmap_r, fmap_g):
                loss = 0

                if self.cfg.model.generator in [
                    "hifigan",
                    "nsfhifigan",
                    "bigvgan",
                    "apnet",
                ]:
                    for dr, dg in zip(fmap_r, fmap_g):
                        for rl, gl in zip(dr, dg):
                            loss += torch.mean(torch.abs(rl - gl))

                    loss = loss * 2
                elif self.cfg.model.generator in ["melgan"]:
                    for dr, dg in zip(fmap_r, fmap_g):
                        for rl, gl in zip(dr, dg):
                            loss += self.l1Loss(rl, gl)

                    loss = loss * 10
                elif self.cfg.model.generator in ["codec"]:
                    for dr, dg in zip(fmap_r, fmap_g):
                        for rl, gl in zip(dr, dg):
                            loss = loss + self.l1Loss(rl, gl) / torch.mean(
                                torch.abs(rl)
                            )

                    KL_scale = len(fmap_r) * len(fmap_r[0])

                    loss = 3 * loss / KL_scale
                else:
                    raise NotImplementedError

                return loss

        class discriminator_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(discriminator_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, disc_real_outputs, disc_generated_outputs):
                loss = 0
                r_losses = []
                g_losses = []

                if self.cfg.model.generator in [
                    "hifigan",
                    "nsfhifigan",
                    "bigvgan",
                    "apnet",
                ]:
                    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
                        r_loss = torch.mean((1 - dr) ** 2)
                        g_loss = torch.mean(dg**2)
                        loss += r_loss + g_loss
                        r_losses.append(r_loss.item())
                        g_losses.append(g_loss.item())
                elif self.cfg.model.generator in ["melgan"]:
                    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
                        r_loss = torch.mean(self.relu(1 - dr))
                        g_loss = torch.mean(self.relu(1 + dg))
                        loss = loss + r_loss + g_loss
                        r_losses.append(r_loss.item())
                        g_losses.append(g_loss.item())
                elif self.cfg.model.generator in ["codec"]:
                    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
                        r_loss = torch.mean(self.relu(1 - dr))
                        g_loss = torch.mean(self.relu(1 + dg))
                        loss = loss + r_loss + g_loss
                        r_losses.append(r_loss.item())
                        g_losses.append(g_loss.item())

                    loss = loss / len(disc_real_outputs)
                else:
                    raise NotImplementedError

                return loss, r_losses, g_losses

        class generator_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(generator_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, disc_outputs):
                loss = 0
                gen_losses = []

                if self.cfg.model.generator in [
                    "hifigan",
                    "nsfhifigan",
                    "bigvgan",
                    "apnet",
                ]:
                    for dg in disc_outputs:
                        l = torch.mean((1 - dg) ** 2)
                        gen_losses.append(l)
                        loss += l
                elif self.cfg.model.generator in ["melgan"]:
                    for dg in disc_outputs:
                        l = -torch.mean(dg)
                        gen_losses.append(l)
                        loss += l
                elif self.cfg.model.generator in ["codec"]:
                    for dg in disc_outputs:
                        l = torch.mean(self.relu(1 - dg)) / len(disc_outputs)
                        gen_losses.append(l)
                        loss += l
                else:
                    raise NotImplementedError

                return loss, gen_losses

        class mel_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(mel_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, y_gt, y_pred):
                loss = 0

                if self.cfg.model.generator in [
                    "hifigan",
                    "nsfhifigan",
                    "bigvgan",
                    "melgan",
                    "codec",
                    "apnet",
                ]:
                    y_gt_mel = extract_mel_features(y_gt, self.cfg.preprocess)
                    y_pred_mel = extract_mel_features(
                        y_pred.squeeze(1), self.cfg.preprocess
                    )

                    loss = self.l1Loss(y_gt_mel, y_pred_mel) * 45
                else:
                    raise NotImplementedError

                return loss

        class wav_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(wav_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, y_gt, y_pred):
                loss = 0

                if self.cfg.model.generator in [
                    "hifigan",
                    "nsfhifigan",
                    "bigvgan",
                    "apnet",
                ]:
                    loss = self.l2Loss(y_gt, y_pred.squeeze(1)) * 100
                elif self.cfg.model.generator in ["melgan"]:
                    loss = self.l1Loss(y_gt, y_pred.squeeze(1)) / 10
                elif self.cfg.model.generator in ["codec"]:
                    loss = self.l1Loss(y_gt, y_pred.squeeze(1)) + self.l2Loss(
                        y_gt, y_pred.squeeze(1)
                    )
                    loss /= 10
                else:
                    raise NotImplementedError

                return loss

        class phase_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(phase_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, phase_gt, phase_pred):
                n_fft = self.cfg.preprocess.n_fft
                frames = phase_gt.size()[-1]

                GD_matrix = (
                    torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=1)
                    - torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=2)
                    - torch.eye(n_fft // 2 + 1)
                )
                GD_matrix = GD_matrix.to(phase_pred.device)

                GD_r = torch.matmul(phase_gt.permute(0, 2, 1), GD_matrix)
                GD_g = torch.matmul(phase_pred.permute(0, 2, 1), GD_matrix)

                PTD_matrix = (
                    torch.triu(torch.ones(frames, frames), diagonal=1)
                    - torch.triu(torch.ones(frames, frames), diagonal=2)
                    - torch.eye(frames)
                )
                PTD_matrix = PTD_matrix.to(phase_pred.device)

                PTD_r = torch.matmul(phase_gt, PTD_matrix)
                PTD_g = torch.matmul(phase_pred, PTD_matrix)

                IP_loss = torch.mean(-torch.cos(phase_gt - phase_pred))
                GD_loss = torch.mean(-torch.cos(GD_r - GD_g))
                PTD_loss = torch.mean(-torch.cos(PTD_r - PTD_g))

                return 100 * (IP_loss + GD_loss + PTD_loss)

        class amplitude_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(amplitude_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(self, log_amplitude_gt, log_amplitude_pred):
                amplitude_loss = self.l2Loss(log_amplitude_gt, log_amplitude_pred)

                return 45 * amplitude_loss

        class consistency_criterion(torch.nn.Module):
            def __init__(self, cfg):
                super(consistency_criterion, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")
                self.l2Loss = torch.nn.MSELoss(reduction="mean")
                self.relu = torch.nn.ReLU()

            def __call__(
                self,
                rea_gt,
                rea_pred,
                rea_pred_final,
                imag_gt,
                imag_pred,
                imag_pred_final,
            ):
                C_loss = torch.mean(
                    torch.mean(
                        (rea_pred - rea_pred_final) ** 2
                        + (imag_pred - imag_pred_final) ** 2,
                        (1, 2),
                    )
                )

                L_R = self.l1Loss(rea_gt, rea_pred)
                L_I = self.l1Loss(imag_gt, imag_pred)

                return 20 * (C_loss + 2.25 * (L_R + L_I))

        criterions = dict()
        for key in self.cfg.train.criterions:
            if key == "feature":
                criterions["feature"] = feature_criterion(self.cfg)
            elif key == "discriminator":
                criterions["discriminator"] = discriminator_criterion(self.cfg)
            elif key == "generator":
                criterions["generator"] = generator_criterion(self.cfg)
            elif key == "mel":
                criterions["mel"] = mel_criterion(self.cfg)
            elif key == "wav":
                criterions["wav"] = wav_criterion(self.cfg)
            elif key == "phase":
                criterions["phase"] = phase_criterion(self.cfg)
            elif key == "amplitude":
                criterions["amplitude"] = amplitude_criterion(self.cfg)
            elif key == "consistency":
                criterions["consistency"] = consistency_criterion(self.cfg)
            else:
                raise NotImplementedError

        return criterions

    def _build_model(self):
        generator = supported_generators[self.cfg.model.generator](self.cfg)
        discriminators = dict()
        for key in self.cfg.model.discriminators:
            discriminators[key] = supported_discriminators[key](self.cfg)

        return generator, discriminators

    def _build_optimizer(self):
        optimizer_params_generator = [dict(params=self.generator.parameters())]
        generator_optimizer = AdamW(
            optimizer_params_generator,
            lr=self.cfg.train.adamw.lr,
            betas=(self.cfg.train.adamw.adam_b1, self.cfg.train.adamw.adam_b2),
        )

        optimizer_params_discriminator = []
        for discriminator in self.discriminators.keys():
            optimizer_params_discriminator.append(
                dict(params=self.discriminators[discriminator].parameters())
            )
        discriminator_optimizer = AdamW(
            optimizer_params_discriminator,
            lr=self.cfg.train.adamw.lr,
            betas=(self.cfg.train.adamw.adam_b1, self.cfg.train.adamw.adam_b2),
        )

        return generator_optimizer, discriminator_optimizer

    def _build_scheduler(self):
        discriminator_scheduler = ExponentialLR(
            self.discriminator_optimizer,
            gamma=self.cfg.train.exponential_lr.lr_decay,
            last_epoch=self.epoch - 1,
        )

        generator_scheduler = ExponentialLR(
            self.generator_optimizer,
            gamma=self.cfg.train.exponential_lr.lr_decay,
            last_epoch=self.epoch - 1,
        )

        return generator_scheduler, discriminator_scheduler

    def train_loop(self):
        """Training process"""
        self.accelerator.wait_for_everyone()

        # Dump config
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)
        self.generator.train()
        for key in self.discriminators.keys():
            self.discriminators[key].train()
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        # Sync and start training
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            # Train and Validate
            train_total_loss, train_losses = self._train_epoch()
            for key, loss in train_losses.items():
                self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                self.accelerator.log(
                    {"Epoch/Train {} Loss".format(key): loss},
                    step=self.epoch,
                )
            valid_total_loss, valid_losses = self._valid_epoch()
            for key, loss in valid_losses.items():
                self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
                self.accelerator.log(
                    {"Epoch/Valid {} Loss".format(key): loss},
                    step=self.epoch,
                )
            self.accelerator.log(
                {
                    "Epoch/Train Total Loss": train_total_loss,
                    "Epoch/Valid Total Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            # Update scheduler
            self.accelerator.wait_for_everyone()
            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            # Check save checkpoint interval
            run_eval = False
            if self.accelerator.is_main_process:
                save_checkpoint = False
                for i, num in enumerate(self.save_checkpoint_stride):
                    if self.epoch % num == 0:
                        save_checkpoint = True
                        run_eval |= self.run_eval[i]

            # Save checkpoints
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and save_checkpoint:
                path = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_total_loss
                    ),
                )
                self.accelerator.save_state(path)
                json.dump(
                    self.checkpoints_path,
                    open(os.path.join(path, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )

            # Save eval audios
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and run_eval:
                for i in range(len(self.valid_dataloader.dataset.eval_audios)):
                    if self.cfg.preprocess.use_frame_pitch:
                        eval_audio = self._inference(
                            self.valid_dataloader.dataset.eval_mels[i],
                            eval_pitch=self.valid_dataloader.dataset.eval_pitchs[i],
                            use_pitch=True,
                        )
                    else:
                        eval_audio = self._inference(
                            self.valid_dataloader.dataset.eval_mels[i]
                        )
                    path = os.path.join(
                        self.checkpoint_dir,
                        "epoch-{:04d}_step-{:07d}_loss-{:.6f}_eval_audio_{}.wav".format(
                            self.epoch,
                            self.step,
                            valid_total_loss,
                            self.valid_dataloader.dataset.eval_dataset_names[i],
                        ),
                    )
                    path_gt = os.path.join(
                        self.checkpoint_dir,
                        "epoch-{:04d}_step-{:07d}_loss-{:.6f}_eval_audio_{}_gt.wav".format(
                            self.epoch,
                            self.step,
                            valid_total_loss,
                            self.valid_dataloader.dataset.eval_dataset_names[i],
                        ),
                    )
                    save_audio(path, eval_audio, self.cfg.preprocess.sample_rate)
                    save_audio(
                        path_gt,
                        self.valid_dataloader.dataset.eval_audios[i],
                        self.cfg.preprocess.sample_rate,
                    )

            self.accelerator.wait_for_everyone()

            self.epoch += 1

        # Finish training
        self.accelerator.wait_for_everyone()
        path = os.path.join(
            self.checkpoint_dir,
            "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, valid_total_loss
            ),
        )
        self.accelerator.save_state(path)

    def _train_epoch(self):
        """Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.generator.train()
        for key, _ in self.discriminators.items():
            self.discriminators[key].train()

        epoch_losses: dict = {}
        epoch_total_loss: int = 0

        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Get losses
            total_loss, losses = self._train_step(batch)
            self.batch_count += 1

            # Log info
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                self.accelerator.log(
                    {
                        "Step/Generator Learning Rate": self.generator_optimizer.param_groups[
                            0
                        ][
                            "lr"
                        ],
                        "Step/Discriminator Learning Rate": self.discriminator_optimizer.param_groups[
                            0
                        ][
                            "lr"
                        ],
                    },
                    step=self.step,
                )
                for key, _ in losses.items():
                    self.accelerator.log(
                        {
                            "Step/Train {} Loss".format(key): losses[key],
                        },
                        step=self.step,
                    )

                if not epoch_losses:
                    epoch_losses = losses
                else:
                    for key, value in losses.items():
                        epoch_losses[key] += value
                epoch_total_loss += total_loss
                self.step += 1

        # Get and log total losses
        self.accelerator.wait_for_everyone()
        epoch_total_loss = (
            epoch_total_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )
        for key in epoch_losses.keys():
            epoch_losses[key] = (
                epoch_losses[key]
                / len(self.train_dataloader)
                * self.cfg.train.gradient_accumulation_step
            )
        return epoch_total_loss, epoch_losses

    def _train_step(self, data):
        """Training forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_train_epoch`` for usage.
        """
        # Init losses
        train_losses = {}
        total_loss = 0

        generator_losses = {}
        generator_total_loss = 0
        discriminator_losses = {}
        discriminator_total_loss = 0

        # Use input feature to get predictions
        mel_input = data["mel"]
        audio_gt = data["audio"]

        if self.cfg.preprocess.extract_amplitude_phase:
            logamp_gt = data["logamp"]
            pha_gt = data["pha"]
            rea_gt = data["rea"]
            imag_gt = data["imag"]

        if self.cfg.preprocess.use_frame_pitch:
            pitch_input = data["frame_pitch"]

        if self.cfg.preprocess.use_frame_pitch:
            pitch_input = pitch_input.float()
            audio_pred = self.generator.forward(mel_input, pitch_input)
        elif self.cfg.preprocess.extract_amplitude_phase:
            (
                logamp_pred,
                pha_pred,
                rea_pred,
                imag_pred,
                audio_pred,
            ) = self.generator.forward(mel_input)
            from utils.mel import amplitude_phase_spectrum

            _, _, rea_pred_final, imag_pred_final = amplitude_phase_spectrum(
                audio_pred.squeeze(1), self.cfg.preprocess
            )
        else:
            audio_pred = self.generator.forward(mel_input)

        # Calculate and BP Discriminator losses
        self.discriminator_optimizer.zero_grad()
        for key, _ in self.discriminators.items():
            y_r, y_g, _, _ = self.discriminators[key].forward(
                audio_gt.unsqueeze(1), audio_pred.detach()
            )
            (
                discriminator_losses["{}_discriminator".format(key)],
                _,
                _,
            ) = self.criterions["discriminator"](y_r, y_g)
            discriminator_total_loss += discriminator_losses[
                "{}_discriminator".format(key)
            ]

        self.accelerator.backward(discriminator_total_loss)
        self.discriminator_optimizer.step()

        # Calculate and BP Generator losses
        self.generator_optimizer.zero_grad()
        for key, _ in self.discriminators.items():
            y_r, y_g, f_r, f_g = self.discriminators[key].forward(
                audio_gt.unsqueeze(1), audio_pred
            )
            generator_losses["{}_feature".format(key)] = self.criterions["feature"](
                f_r, f_g
            )
            generator_losses["{}_generator".format(key)], _ = self.criterions[
                "generator"
            ](y_g)
            generator_total_loss += generator_losses["{}_feature".format(key)]
            generator_total_loss += generator_losses["{}_generator".format(key)]

        if "mel" in self.criterions.keys():
            generator_losses["mel"] = self.criterions["mel"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["mel"]

        if "wav" in self.criterions.keys():
            generator_losses["wav"] = self.criterions["wav"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["wav"]

        if "amplitude" in self.criterions.keys():
            generator_losses["amplitude"] = self.criterions["amplitude"](
                logamp_gt, logamp_pred
            )
            generator_total_loss += generator_losses["amplitude"]

        if "phase" in self.criterions.keys():
            generator_losses["phase"] = self.criterions["phase"](pha_gt, pha_pred)
            generator_total_loss += generator_losses["phase"]

        if "consistency" in self.criterions.keys():
            generator_losses["consistency"] = self.criterions["consistency"](
                rea_gt, rea_pred, rea_pred_final, imag_gt, imag_pred, imag_pred_final
            )
            generator_total_loss += generator_losses["consistency"]

        self.accelerator.backward(generator_total_loss)
        self.generator_optimizer.step()

        # Get the total losses
        total_loss = discriminator_total_loss + generator_total_loss
        train_losses.update(discriminator_losses)
        train_losses.update(generator_losses)

        for key, _ in train_losses.items():
            train_losses[key] = train_losses[key].item()

        return total_loss.item(), train_losses

    def _valid_epoch(self):
        """Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.generator.eval()
        for key, _ in self.discriminators.items():
            self.discriminators[key].eval()

        epoch_losses: dict = {}
        epoch_total_loss: int = 0

        for batch in tqdm(
            self.valid_dataloader,
            desc=f"Validating Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Get losses
            total_loss, losses = self._valid_step(batch)

            # Log info
            for key, _ in losses.items():
                self.accelerator.log(
                    {
                        "Step/Valid {} Loss".format(key): losses[key],
                    },
                    step=self.step,
                )

            if not epoch_losses:
                epoch_losses = losses
            else:
                for key, value in losses.items():
                    epoch_losses[key] += value
            epoch_total_loss += total_loss

        # Get and log total losses
        self.accelerator.wait_for_everyone()
        epoch_total_loss = epoch_total_loss / len(self.valid_dataloader)
        for key in epoch_losses.keys():
            epoch_losses[key] = epoch_losses[key] / len(self.valid_dataloader)
        return epoch_total_loss, epoch_losses

    def _valid_step(self, data):
        """Testing forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_test_epoch`` for usage.
        """
        # Init losses
        valid_losses = {}
        total_loss = 0

        generator_losses = {}
        generator_total_loss = 0
        discriminator_losses = {}
        discriminator_total_loss = 0

        # Use feature inputs to get the predicted audio
        mel_input = data["mel"]
        audio_gt = data["audio"]

        if self.cfg.preprocess.extract_amplitude_phase:
            logamp_gt = data["logamp"]
            pha_gt = data["pha"]
            rea_gt = data["rea"]
            imag_gt = data["imag"]

        if self.cfg.preprocess.use_frame_pitch:
            pitch_input = data["frame_pitch"]

        if self.cfg.preprocess.use_frame_pitch:
            pitch_input = pitch_input.float()
            audio_pred = self.generator.forward(mel_input, pitch_input)
        elif self.cfg.preprocess.extract_amplitude_phase:
            (
                logamp_pred,
                pha_pred,
                rea_pred,
                imag_pred,
                audio_pred,
            ) = self.generator.forward(mel_input)
            from utils.mel import amplitude_phase_spectrum

            _, _, rea_pred_final, imag_pred_final = amplitude_phase_spectrum(
                audio_pred.squeeze(1), self.cfg.preprocess
            )
        else:
            audio_pred = self.generator.forward(mel_input)

        # Get Discriminator losses
        for key, _ in self.discriminators.items():
            y_r, y_g, _, _ = self.discriminators[key].forward(
                audio_gt.unsqueeze(1), audio_pred
            )
            (
                discriminator_losses["{}_discriminator".format(key)],
                _,
                _,
            ) = self.criterions["discriminator"](y_r, y_g)
            discriminator_total_loss += discriminator_losses[
                "{}_discriminator".format(key)
            ]

        for key, _ in self.discriminators.items():
            y_r, y_g, f_r, f_g = self.discriminators[key].forward(
                audio_gt.unsqueeze(1), audio_pred
            )
            generator_losses["{}_feature".format(key)] = self.criterions["feature"](
                f_r, f_g
            )
            generator_losses["{}_generator".format(key)], _ = self.criterions[
                "generator"
            ](y_g)
            generator_total_loss += generator_losses["{}_feature".format(key)]
            generator_total_loss += generator_losses["{}_generator".format(key)]

        if "mel" in self.criterions.keys():
            generator_losses["mel"] = self.criterions["mel"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["mel"]
        if "mel" in self.criterions.keys():
            generator_losses["mel"] = self.criterions["mel"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["mel"]

        if "wav" in self.criterions.keys():
            generator_losses["wav"] = self.criterions["wav"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["wav"]
        if "wav" in self.criterions.keys():
            generator_losses["wav"] = self.criterions["wav"](audio_gt, audio_pred)
            generator_total_loss += generator_losses["wav"]

        if "amplitude" in self.criterions.keys():
            generator_losses["amplitude"] = self.criterions["amplitude"](
                logamp_gt, logamp_pred
            )
            generator_total_loss += generator_losses["amplitude"]

        if "phase" in self.criterions.keys():
            generator_losses["phase"] = self.criterions["phase"](pha_gt, pha_pred)
            generator_total_loss += generator_losses["phase"]

        if "consistency" in self.criterions.keys():
            generator_losses["consistency"] = self.criterions["consistency"](
                rea_gt,
                rea_pred,
                rea_pred_final,
                imag_gt,
                imag_pred,
                imag_pred_final,
            )
            generator_total_loss += generator_losses["consistency"]

        total_loss = discriminator_total_loss + generator_total_loss
        valid_losses.update(discriminator_losses)
        valid_losses.update(generator_losses)

        for item in valid_losses:
            valid_losses[item] = valid_losses[item].item()

        return total_loss.item(), valid_losses

    def _inference(self, eval_mel, eval_pitch=None, use_pitch=False):
        """Inference during training for test audios."""
        if use_pitch:
            eval_pitch = align_length(eval_pitch, eval_mel.shape[1])
            eval_audio = vocoder_inference(
                self.cfg,
                self.generator,
                torch.from_numpy(eval_mel).unsqueeze(0),
                f0s=torch.from_numpy(eval_pitch).unsqueeze(0).float(),
                device=next(self.generator.parameters()).device,
            ).squeeze(0)
        else:
            eval_audio = vocoder_inference(
                self.cfg,
                self.generator,
                torch.from_numpy(eval_mel).unsqueeze(0),
                device=next(self.generator.parameters()).device,
            ).squeeze(0)
        return eval_audio

    def _load_model(self, checkpoint_dir, checkpoint_path=None, resume_type="resume"):
        """Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            ls = [str(i) for i in Path(checkpoint_dir).glob("*")]
            ls.sort(key=lambda x: int(x.split("_")[-3].split("-")[-1]), reverse=True)
            checkpoint_path = ls[0]
        if resume_type == "resume":
            self.accelerator.load_state(checkpoint_path)
            self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1]) + 1
            self.step = int(checkpoint_path.split("_")[-2].split("-")[-1]) + 1
        elif resume_type == "finetune":
            accelerate.load_checkpoint_and_dispatch(
                self.accelerator.unwrap_model(self.generator),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
            )
            for key, _ in self.discriminators.items():
                accelerate.load_checkpoint_and_dispatch(
                    self.accelerator.unwrap_model(self.discriminators[key]),
                    os.path.join(checkpoint_path, "pytorch_model.bin"),
                )
            self.logger.info("Load model weights for finetune SUCCESS!")
        else:
            raise ValueError("Unsupported resume type: {}".format(resume_type))
        return checkpoint_path

    def _count_parameters(self):
        result = sum(p.numel() for p in self.generator.parameters())
        for _, discriminator in self.discriminators.items():
            result += sum(p.numel() for p in discriminator.parameters())
        return result
