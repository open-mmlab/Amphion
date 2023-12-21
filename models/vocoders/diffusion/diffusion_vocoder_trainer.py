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
import torch.nn as nn
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
from models.vocoders.diffusion.diffusion_vocoder_dataset import (
    DiffusionVocoderDataset,
    DiffusionVocoderCollator,
)

from models.vocoders.diffusion.diffwave.diffwave import DiffWave

from models.vocoders.diffusion.diffusion_vocoder_inference import vocoder_inference

supported_models = {
    "diffwave": DiffWave,
}


class DiffusionVocoderTrainer(VocoderTrainer):
    def __init__(self, args, cfg):
        super().__init__()

        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        # Diffusion
        self.cfg.model.diffwave.noise_schedule = np.linspace(
            self.cfg.model.diffwave.noise_schedule_factors[0],
            self.cfg.model.diffwave.noise_schedule_factors[1],
            self.cfg.model.diffwave.noise_schedule_factors[2],
        )
        beta = np.array(self.cfg.model.diffwave.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

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
            self.model = self._build_model()
            end = time.monotonic_ns()
            self.logger.debug(self.model)
            self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
            self.logger.info(f"Model parameters: {self._count_parameters()/1e6:.2f}M")

        # Build optimizers and schedulers
        with self.accelerator.main_process_first():
            self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
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
            self.model,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.valid_dataloader,
            self.model,
            self.optimizer,
            self.scheduler,
        )
        end = time.monotonic_ns()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # Build criterions
        with self.accelerator.main_process_first():
            self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
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

        # Device
        self.device = next(self.model.parameters()).device
        self.noise_level = self.noise_level.to(self.device)

    def _build_dataset(self):
        return DiffusionVocoderDataset, DiffusionVocoderCollator

    def _build_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def _build_model(self):
        model = supported_models[self.cfg.model.generator](self.cfg)
        return model

    def _build_optimizer(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.train.adamw.lr,
            betas=(self.cfg.train.adamw.adam_b1, self.cfg.train.adamw.adam_b2),
        )
        return optimizer

    def _build_scheduler(self):
        scheduler = ExponentialLR(
            self.optimizer,
            gamma=self.cfg.train.exponential_lr.lr_decay,
            last_epoch=self.epoch - 1,
        )
        return scheduler

    def train_loop(self):
        """Training process"""
        self.accelerator.wait_for_everyone()

        # Dump config
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)
        self.model.train()
        self.optimizer.zero_grad()

        # Sync and start training
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            # Train and Validate
            train_total_loss = self._train_epoch()
            valid_total_loss = self._valid_epoch()
            self.accelerator.log(
                {
                    "Epoch/Train Total Loss": train_total_loss,
                    "Epoch/Valid Total Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            # Update scheduler
            self.accelerator.wait_for_everyone()
            self.scheduler.step()

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
        self.model.train()

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
            total_loss = self._train_step(batch)
            self.batch_count += 1

            # Log info
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                self.accelerator.log(
                    {
                        "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.step,
                )
                epoch_total_loss += total_loss
                self.step += 1

        # Get and log total losses
        self.accelerator.wait_for_everyone()
        epoch_total_loss = (
            epoch_total_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )
        return epoch_total_loss

    def _train_step(self, data):
        """Training forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_train_epoch`` for usage.
        """
        # Init losses
        total_loss = 0

        # Use input feature to get predictions
        mel_input = data["mel"]
        audio_gt = data["audio"]

        if self.cfg.preprocess.use_frame_pitch:
            pitch_input = data["frame_pitch"]

        self.optimizer.zero_grad()
        N = audio_gt.shape[0]
        t = torch.randint(
            0, len(self.cfg.model.diffwave.noise_schedule), [N], device=self.device
        )
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio_gt).to(self.device)
        noisy_audio = noise_scale_sqrt * audio_gt + (1.0 - noise_scale) ** 0.5 * noise

        audio_pred = self.model(noisy_audio, t, mel_input)
        total_loss = self.criterion(noise, audio_pred.squeeze(1))

        self.accelerator.backward(total_loss)
        self.optimizer.step()

        return total_loss.item()

    def _valid_epoch(self):
        """Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.eval()

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
            total_loss = self._valid_step(batch)

            # Log info
            epoch_total_loss += total_loss

        # Get and log total losses
        self.accelerator.wait_for_everyone()
        epoch_total_loss = epoch_total_loss / len(self.valid_dataloader)
        return epoch_total_loss

    def _valid_step(self, data):
        """Testing forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_test_epoch`` for usage.
        """
        # Init losses
        total_loss = 0

        # Use feature inputs to get the predicted audio
        mel_input = data["mel"]
        audio_gt = data["audio"]

        if self.cfg.preprocess.use_frame_pitch:
            pitch_input = data["frame_pitch"]

        N = audio_gt.shape[0]
        t = torch.randint(
            0, len(self.cfg.model.diffwave.noise_schedule), [N], device=self.device
        )
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio_gt)
        noisy_audio = noise_scale_sqrt * audio_gt + (1.0 - noise_scale) ** 0.5 * noise

        audio_pred = self.model(noisy_audio, t, mel_input)
        total_loss = self.criterion(noise, audio_pred.squeeze(1))

        return total_loss.item()

    def _inference(self, eval_mel, eval_pitch=None, use_pitch=False):
        """Inference during training for test audios."""
        if use_pitch:
            eval_pitch = align_length(eval_pitch, eval_mel.shape[1])
            eval_audio = vocoder_inference(
                self.cfg,
                self.model,
                torch.from_numpy(eval_mel).unsqueeze(0),
                f0s=torch.from_numpy(eval_pitch).unsqueeze(0).float(),
                device=next(self.model.parameters()).device,
            ).squeeze(0)
        else:
            eval_audio = vocoder_inference(
                self.cfg,
                self.model,
                torch.from_numpy(eval_mel).unsqueeze(0),
                device=next(self.model.parameters()).device,
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
                self.accelerator.unwrap_model(self.model),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
            )
            self.logger.info("Load model weights for finetune SUCCESS!")
        else:
            raise ValueError("Unsupported resume type: {}".format(resume_type))
        return checkpoint_path

    def _count_parameters(self):
        result = sum(p.numel() for p in self.model.parameters())
        return result
