# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json
import time
import torch
import numpy as np
import math
from utils.util import Logger, ValueWindow
from torch.utils.data import ConcatDataset, DataLoader

from models.tts.base.tts_trainer import TTSTrainer
from models.base.base_trainer import BaseTrainer
from models.base.base_sampler import VariableSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.optim import Adam, AdamW
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from models.codec.melvqgan.melspec import MelSpectrogram
from transformers import get_inverse_sqrt_schedule, get_constant_schedule

import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from models.codec.amphion_codec.vocos import Vocos
from models.codec.amphion_codec.loss import (
    MultiResolutionSTFTLoss,
    MultiResolutionMelSpectrogramLoss,
    GANLoss,
)
from models.codec.discriminator.hifigan_disriminator import (
    HiFiGANMultiPeriodDiscriminator,
    SpecDiscriminator,
)

from itertools import chain
from models.codec.coco.coco_dataset import CocoCollator
from models.vocoders.vocos.vocos_dataset import VocosDataset


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


class VocosTrainer(TTSTrainer):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                os.makedirs(os.path.join(self.exp_dir, "checkpoint"), exist_ok=True)
                self.log_file = os.path.join(
                    os.path.join(self.exp_dir, "checkpoint"), "train.log"
                )
                self.logger = Logger(self.log_file, level=self.args.log_level).logger

        self.time_window = ValueWindow(100)

        if self.accelerator.is_main_process:
            # Log some info
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

        if self.accelerator.is_main_process:
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(
                    f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
                )
                self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )

        # setup model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(self.model)
                self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
                self.logger.info(
                    f"Vocoder parameters: {self._count_parameters(self.model['vocoder'])/1e6:.2f}M"
                )
                self.logger.info(
                    f"Period GAN parameters: {self._count_parameters(self.model['period_gan'])/1e6:.2f}M"
                )
                self.logger.info(
                    f"Spec GAN parameters: {self._count_parameters(self.model['spec_gan'])/1e6:.2f}M"
                )

        # setup mel model
        self._build_mel_model()

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
                )

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            if self.accelerator.is_main_process:
                self.logger.info("Initializing accelerate...")
            start = time.monotonic_ns()
            self.train_dataloader = self.accelerator.prepare(
                self.train_dataloader,
            )

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        end = time.monotonic_ns()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criteria = self._build_criterion()
            for key in self.criteria.keys():
                self.criteria[key] = self.criteria[key].to(self.accelerator.device)
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # Resume or Finetune
        with self.accelerator.main_process_first():
            if args.resume:
                ## Automatically resume according to the current exprimental name
                print(
                    "Automatically resuming from latest checkpoint in {}...".format(
                        self.checkpoint_dir
                    )
                )
                start = time.monotonic_ns()
                ckpt_path = self._load_model(
                    checkpoint_dir=self.checkpoint_dir, resume_type=args.resume_type
                )
                end = time.monotonic_ns()
                print(f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        # Only for TTS tasks
        self.task_type = "TTS"
        if self.accelerator.is_main_process:
            self.logger.info("Task type: {}".format(self.task_type))

    def _count_parameters(self, model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            # kwargs_handlers=[ddp_kwargs]
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def _build_model(self):
        vocoder = Vocos(cfg=self.cfg.model.vocos)
        period_gan = HiFiGANMultiPeriodDiscriminator(cfg=self.cfg.model.period_gan)
        spec_gan = SpecDiscriminator(cfg=self.cfg.model.spec_gan)
        return {
            "vocoder": vocoder,
            "period_gan": period_gan,
            "spec_gan": spec_gan,
        }

    def _build_mel_model(self):
        self.mel_model = MelSpectrogram(
            sampling_rate=self.cfg.preprocess.sample_rate,
            n_fft=self.cfg.preprocess.n_fft,
            num_mels=self.cfg.preprocess.num_mels,
            hop_size=self.cfg.preprocess.hop_size,
            win_size=self.cfg.preprocess.win_size,
            fmin=self.cfg.preprocess.fmin,
            fmax=self.cfg.preprocess.fmax,
        )
        self.mel_model.eval()
        self.mel_model.to(self.accelerator.device)

    def _build_dataset(self):
        return VocosDataset, CocoCollator

    def _build_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize......")
            Dataset, Collator = self._build_dataset()
            if (
                hasattr(self.cfg.train, "use_emilia_dataset")
                and self.cfg.train.use_emilia_dataset
            ):
                train_dataset = Dataset(cfg=self.cfg)
            else:
                train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)
            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            np.random.seed(111)
            np.random.shuffle(batch_sampler)
            print(batch_sampler[:1])
            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]

            train_loader = DataLoader(
                train_dataset,
                collate_fn=train_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False, use_random_sampler=True
                ),
                pin_memory=self.cfg.train.dataloader.pin_memory,
                prefetch_factor=32,
            )
            self.accelerator.wait_for_everyone()

            valid_loader = None

        else:
            print("Use Normal Batchsize......")
            Dataset, Collator = self._build_dataset()
            if (
                hasattr(self.cfg.train, "use_emilia_dataset")
                and self.cfg.train.use_emilia_dataset
            ):
                train_dataset = Dataset(cfg=self.cfg)
            else:
                train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=train_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
                prefetch_factor=32,
            )

            valid_loader = None
            self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

    def _build_optimizer(self):
        params_g = self.model["vocoder"].parameters()
        optimizer_g = torch.optim.Adam(
            params_g,
            **self.cfg.train.adam_g,
        )
        params_d = self.model["period_gan"].parameters()
        params_d = chain(params_d, self.model["spec_gan"].parameters())
        optimizer_d = torch.optim.Adam(
            params_d,
            **self.cfg.train.adam_d,
        )
        optimizer = {"optimizer_g": optimizer_g, "optimizer_d": optimizer_d}
        return optimizer

    def _build_scheduler(self):
        scheduler_g = get_constant_schedule(self.optimizer["optimizer_g"])
        scheduler_d = get_constant_schedule(self.optimizer["optimizer_d"])

        scheduler = {"scheduler_g": scheduler_g, "scheduler_d": scheduler_d}
        return scheduler

    def _build_criterion(self):
        criteria = dict()
        criteria["gan_loss"] = GANLoss(mode="lsgan")
        criteria["mel_loss"] = MultiResolutionMelSpectrogramLoss(
            cfg=self.cfg.loss.mel_loss
        )
        criteria["fm_loss"] = torch.nn.L1Loss()
        return criteria

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def get_state_dict(self):
        state_dict = {
            "vocoder": self.model["vocoder"].state_dict(),
            "period_gan": self.model["period_gan"].state_dict(),
            "spec_gan": self.model["spec_gan"].state_dict(),
            "optimizer_g": self.optimizer["optimizer_g"].state_dict(),
            "optimizer_d": self.optimizer["optimizer_d"].state_dict(),
            "scheduler_g": self.scheduler["scheduler_g"].state_dict(),
            "scheduler_d": self.scheduler["scheduler_d"].state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model["vocoder"].load_state_dict(checkpoint["vocoder"])
        self.model["period_gan"].load_state_dict(checkpoint["period_gan"])
        self.model["spec_gan"].load_state_dict(checkpoint["spec_gan"])
        self.optimizer["optimizer_g"].load_state_dict(checkpoint["optimizer_g"])
        self.optimizer["optimizer_d"].load_state_dict(checkpoint["optimizer_d"])
        self.scheduler["scheduler_g"].load_state_dict(checkpoint["scheduler_g"])
        self.scheduler["scheduler_d"].load_state_dict(checkpoint["scheduler_d"])

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        # speech = batch["speech"]  # (B, T)
        speech = batch["wav"]

        with torch.no_grad():
            # mel_feat = self.mel_model(speech)
            mel_feat = (
                self.mel_model(speech) - self.cfg.preprocess.mel_mean
            ) / math.sqrt(self.cfg.preprocess.mel_var)

        y_ = self.model["vocoder"](mel_feat)  # (B, 1, T)
        # y = batch["speech"].unsqueeze(1)
        y = batch["wav"].unsqueeze(1)

        # Discriminator loss, BP, and BP and Grad Updated
        disc_loss, train_losses = self._train_disc_step(y, y_, train_losses)
        self.optimizer["optimizer_d"].zero_grad()
        self.accelerator.backward(disc_loss)
        self.optimizer["optimizer_d"].step()
        self.scheduler["scheduler_d"].step()

        total_loss += disc_loss

        # Generator loss, BP, and BP and Grad Updated

        gen_loss, train_losses = self._train_gen_step(y, y_, train_losses)
        self.optimizer["optimizer_g"].zero_grad()
        self.accelerator.backward(gen_loss)
        self.optimizer["optimizer_g"].step()
        self.scheduler["scheduler_g"].step()

        total_loss += gen_loss

        self.current_loss = total_loss.item()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        train_losses["batch_size"] = speech.size(0)
        # learning rate
        train_losses["learning_rate"] = self.optimizer["optimizer_g"].param_groups[0][
            "lr"
        ]

        return total_loss, train_losses, train_stats

    def _train_disc_step(self, y, y_, train_losses):
        # period discriminator
        p = self.model["period_gan"](y)
        p_ = self.model["period_gan"](y_.detach())

        real_loss_list = []
        fake_loss_list = []

        for i in range(len(p)):
            real_loss, fake_loss = self.criteria["gan_loss"].disc_loss(
                p[i][-1], p_[i][-1]
            )
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        # spec discriminator
        sd_p = self.model["spec_gan"](y)
        sd_p_ = self.model["spec_gan"](y_.detach())

        for i in range(len(sd_p)):
            real_loss, fake_loss = self.criteria["gan_loss"].disc_loss(
                sd_p[i][-1], sd_p_[i][-1]
            )
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = disc_loss * self.cfg.loss.disc_loss_weight

        train_losses["disc_loss"] = disc_loss
        train_losses["real_loss"] = real_loss
        train_losses["fake_loss"] = fake_loss

        return disc_loss, train_losses

    def _train_gen_step(self, y, y_, train_losses):
        gen_loss = 0.0

        # set discriminator to eval mode
        for param in self.model["period_gan"].parameters():
            param.requires_grad = False
        for param in self.model["spec_gan"].parameters():
            param.requires_grad = False

        # mel loss
        mel_loss = self.criteria["mel_loss"](y_.squeeze(1), y.squeeze(1))
        gen_loss += mel_loss * self.cfg.loss.mel_loss_weight
        train_losses["mel_loss"] = mel_loss

        # gan loss
        p_ = self.model["period_gan"](y_)
        adv_loss_list = []
        for i in range(len(p_)):
            adv_loss_list.append(self.criteria["gan_loss"].gen_loss(p_[i][-1]))

        sd_p_ = self.model["spec_gan"](y_)
        for i in range(len(sd_p_)):
            adv_loss_list.append(self.criteria["gan_loss"].gen_loss(sd_p_[i][-1]))

        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * self.cfg.loss.adv_loss_weight
        train_losses["adv_loss"] = adv_loss

        # feature matching loss
        fm_loss = 0.0
        with torch.no_grad():
            p = self.model["period_gan"](y)
        for i in range(len(p_)):
            for j in range(len(p_[i]) - 1):
                fm_loss += self.criteria["fm_loss"](p_[i][j], p[i][j].detach())

        gen_loss += fm_loss * self.cfg.loss.fm_loss_weight
        train_losses["fm_loss"] = fm_loss

        # spec feature matching loss
        spec_fm_loss = 0.0
        with torch.no_grad():
            sd_p = self.model["spec_gan"](y)
        for i in range(len(sd_p_)):
            for j in range(len(sd_p_[i]) - 1):
                spec_fm_loss += self.criteria["fm_loss"](
                    sd_p_[i][j], sd_p[i][j].detach()
                )

        gen_loss += spec_fm_loss * self.cfg.loss.spec_fm_loss_weight
        train_losses["spec_fm_loss"] = spec_fm_loss

        # set discriminator to train mode
        for param in self.model["period_gan"].parameters():
            param.requires_grad = True
        for param in self.model["spec_gan"].parameters():
            param.requires_grad = True

        return gen_loss, train_losses

    @torch.inference_mode()
    def _valid_step(self, batch): ...

    @torch.inference_mode()
    def _valid_epoch(self): ...

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
        ema_loss = None

        for batch in self.train_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1
            ema_loss = (
                0.98 * ema_loss + 0.02 * self.current_loss
                if ema_loss is not None
                else self.current_loss
            )
            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss = total_loss
                for key, value in train_losses.items():
                    epoch_losses[key] = value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                if (
                    self.accelerator.is_main_process
                    and self.batch_count
                    % (10 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

                if self.step % self.cfg.train.save_checkpoints_steps == 0:
                    self.save_checkpoint()

                if self.accelerator.is_main_process:
                    if self.step % 100 == 0:
                        print(f"EMA Loss: {ema_loss:.6f}")

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            keep_last = self.keep_last[0]
            # 读取self.checkpoint_dir所有的folder
            all_ckpts = os.listdir(self.checkpoint_dir)

            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            all_ckpts = list(all_ckpts)
            if len(all_ckpts) > keep_last:
                # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                all_ckpts = sorted(
                    all_ckpts, key=lambda x: int(x.split("_")[1].split("-")[1])
                )
                for ckpt in all_ckpts[:-keep_last]:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))
            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            self.logger.info("Saving state to {}...".format(path))
            self.accelerator.save_state(path)
            self.logger.info("Finished saving state.")

    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        # if self.accelerator.is_main_process:
        #     self._dump_cfg(self.config_save_path)

        # self.optimizer.zero_grad()

        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            if self.accelerator.is_main_process:
                self.logger.info("\n")
                self.logger.info("-" * 32)
                self.logger.info("Epoch {}: ".format(self.epoch))

            # Do training & validating epoch
            train_total_loss, train_losses = self._train_epoch()
            if isinstance(train_losses, dict):
                for key, loss in train_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            valid_total_loss, valid_losses = 0.0, 0.0
            # if isinstance(valid_losses, dict):
            #     for key, loss in valid_losses.items():
            #         if self.accelerator.is_main_process:
            #             self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
            #         self.accelerator.log(
            #             {"Epoch/Train {} Loss".format(key): loss},
            #             step=self.epoch,
            #         )

            if self.accelerator.is_main_process:
                self.logger.info("  |- Train/Loss: {:.6f}".format(train_total_loss))
                self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_total_loss))
            self.accelerator.log(
                {
                    "Epoch/Train Loss": train_total_loss,
                    "Epoch/Valid Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_total_loss
                    ),
                )
            )
        self.accelerator.end_training()

    def echo_log(self, losses, mode="Training"):
        message = [
            "{} - Epoch {} Step {}: [{:.3f} s/step]".format(
                mode, self.epoch + 1, self.step, self.time_window.average
            )
        ]

        for key in sorted(losses.keys()):
            if isinstance(losses[key], dict):
                for k, v in losses[key].items():
                    message.append(
                        str(k).split("/")[-1] + "=" + str(round(float(v), 5))
                    )
            else:
                message.append(
                    str(key).split("/")[-1] + "=" + str(round(float(losses[key]), 5))
                )

        message.append("current_loss=" + str(round(float(self.current_loss), 5)))

        self.logger.info(", ".join(message))
