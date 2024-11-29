# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import time
import json5
import torch
import numpy as np
from tqdm import tqdm
from utils.util import ValueWindow
from torch.utils.data import DataLoader
from models.vc.Noro.noro_base_trainer import Noro_base_Trainer
from torch.nn import functional as F
from models.base.base_sampler import VariableSampler

from diffusers import get_scheduler
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from models.vc.Noro.noro_model import Noro_VCmodel
from models.vc.Noro.noro_dataset import VCCollator, VCDataset, batch_by_size
from processors.content_extractor import HubertExtractor
from models.vc.Noro.noro_loss import diff_loss, ConstractiveSpeakerLoss
from utils.mel import mel_spectrogram_torch
from utils.f0 import get_f0_features_using_dio, interpolate
from torch.nn.utils.rnn import pad_sequence


class NoroTrainer(Noro_base_Trainer):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        cfg.exp_name = args.exp_name
        self.content_extractor = "mhubert"

        # Initialize accelerator and ensure all processes are ready
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Initialize logger on the main process
        if self.accelerator.is_main_process:
            self.logger = get_logger(args.exp_name, log_level="INFO")

        # Configure noise and speaker usage
        self.use_ref_noise = self.cfg.trans_exp.use_ref_noise

        # Log configuration on the main process
        if self.accelerator.is_main_process:
            self.logger.info(f"use_ref_noise: {self.use_ref_noise}")

        # Initialize a time window for monitoring metrics
        self.time_window = ValueWindow(50)

        # Log the start of training
        if self.accelerator.is_main_process:
            self.logger.info("=" * 56)
            self.logger.info("||\t\tNew training process started.\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")

        # Initialize checkpoint directory
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # Initialize training counters
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Max epoch: {self.max_epoch if self.max_epoch < float('inf') else 'Unlimited'}"
            )

        # Check basic configuration
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # Set random seed
        with self.accelerator.main_process_first():
            self._set_random_seed(self.cfg.train.random_seed)

        # Setup data loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            self.train_dataloader = self._build_dataloader()
            self.speaker_num = len(self.train_dataloader.dataset.speaker2id)
            if self.accelerator.is_main_process:
                self.logger.info("Speaker num: {}".format(self.speaker_num))

        # Build model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            self.model, self.w2v = self._build_model()

        # Resume training if specified
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Resume training: {}".format(args.resume))
            if args.resume:
                if self.accelerator.is_main_process:
                    self.logger.info("Resuming from checkpoint...")
                ckpt_path = self._load_model(
                    self.checkpoint_dir,
                    args.checkpoint_path,
                    resume_type=args.resume_type,
                )
            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.accelerator.is_main_process:
                self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # Initialize optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()

        # Prepare model, w2v, optimizer, and scheduler for accelerator
        self.model = self._prepare_for_accelerator(self.model)
        self.w2v = self._prepare_for_accelerator(self.w2v)
        self.optimizer = self._prepare_for_accelerator(self.optimizer)
        self.scheduler = self._prepare_for_accelerator(self.scheduler)

        # Build criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            self.criterion = self._build_criterion()

        self.config_save_path = os.path.join(self.exp_dir, "args.json")
        self.task_type = "VC"
        self.contrastive_speaker_loss = ConstractiveSpeakerLoss()

        if self.accelerator.is_main_process:
            self.logger.info("Task type: {}".format(self.task_type))

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def _build_model(self):
        w2v = HubertExtractor(self.cfg)
        model = Noro_VCmodel(cfg=self.cfg.model, use_ref_noise=self.use_ref_noise)
        return model, w2v

    def _build_dataloader(self):
        np.random.seed(int(time.time()))
        if self.accelerator.is_main_process:
            self.logger.info("Use Dynamic Batchsize...")
        train_dataset = VCDataset(self.cfg.trans_exp)
        train_collate = VCCollator(self.cfg)
        batch_sampler = batch_by_size(
            train_dataset.num_frame_indices,
            train_dataset.get_num_frames,
            max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
            max_sentences=self.cfg.train.max_sentences * self.accelerator.num_processes,
            required_batch_size_multiple=self.accelerator.num_processes,
        )
        np.random.shuffle(batch_sampler)
        batches = [
            x[self.accelerator.local_process_index :: self.accelerator.num_processes]
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
        )
        self.accelerator.wait_for_everyone()
        return train_loader

    def _build_optimizer(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.train.adam,
        )
        return optimizer

    def _build_scheduler(self):
        lr_scheduler = get_scheduler(
            self.cfg.train.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.train.lr_warmup_steps,
            num_training_steps=self.cfg.train.num_train_steps,
        )
        return lr_scheduler

    def _build_criterion(self):
        criterion = torch.nn.L1Loss(reduction="mean")
        return criterion

    def _dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _prepare_for_accelerator(self, component):
        if isinstance(component, dict):
            for key in component.keys():
                component[key] = self.accelerator.prepare(component[key])
        else:
            component = self.accelerator.prepare(component)
        return component

    def _train_step(self, batch):
        total_loss = 0.0
        train_losses = {}
        device = self.accelerator.device

        # Move all Tensor data to the specified device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        speech = batch["speech"]
        ref_speech = batch["ref_speech"]

        with torch.set_grad_enabled(False):
            # Extract features and spectrograms
            mel = mel_spectrogram_torch(speech, self.cfg).transpose(1, 2)
            ref_mel = mel_spectrogram_torch(ref_speech, self.cfg).transpose(1, 2)
            mask = batch["mask"]
            ref_mask = batch["ref_mask"]

            # Extract pitch and content features
            audio = speech.cpu().numpy()
            f0s = []
            for i in range(audio.shape[0]):
                wav = audio[i]
                f0 = get_f0_features_using_dio(wav, self.cfg.preprocess)
                f0, _ = interpolate(f0)
                frame_num = len(wav) // self.cfg.preprocess.hop_size
                f0 = torch.from_numpy(f0[:frame_num]).to(speech.device)
                f0s.append(f0)

            pitch = pad_sequence(f0s, batch_first=True, padding_value=0).float()
            pitch = (pitch - pitch.mean(dim=1, keepdim=True)) / (
                pitch.std(dim=1, keepdim=True) + 1e-6
            )  # Normalize pitch (B,T)
            _, content_feature = self.w2v.extract_content_features(
                speech
            )  # semantic (B, T, 768)

            if self.use_ref_noise:
                noisy_ref_mel = mel_spectrogram_torch(
                    batch["noisy_ref_speech"], self.cfg
                ).transpose(1, 2)

        if self.use_ref_noise:
            diff_out, (ref_emb, noisy_ref_emb), (cond_emb, _) = self.model(
                x=mel,
                content_feature=content_feature,
                pitch=pitch,
                x_ref=ref_mel,
                x_mask=mask,
                x_ref_mask=ref_mask,
                noisy_x_ref=noisy_ref_mel,
            )
        else:
            diff_out, (ref_emb, _), (cond_emb, _) = self.model(
                x=mel,
                content_feature=content_feature,
                pitch=pitch,
                x_ref=ref_mel,
                x_mask=mask,
                x_ref_mask=ref_mask,
            )

        if self.use_ref_noise:
            # B x N_query x D
            ref_emb = torch.mean(ref_emb, dim=1)  # B x D
            noisy_ref_emb = torch.mean(noisy_ref_emb, dim=1)  # B x D
            all_ref_emb = torch.cat([ref_emb, noisy_ref_emb], dim=0)  # 2B x D
            all_speaker_ids = torch.cat(
                [batch["speaker_id"], batch["speaker_id"]], dim=0
            )  # 2B
            cs_loss = self.contrastive_speaker_loss(all_ref_emb, all_speaker_ids) * 0.25
            total_loss += cs_loss
            train_losses["ref_loss"] = cs_loss

        diff_loss_x0 = diff_loss(diff_out["x0_pred"], mel, mask=mask)
        total_loss += diff_loss_x0
        train_losses["diff_loss_x0"] = diff_loss_x0

        diff_loss_noise = diff_loss(
            diff_out["noise_pred"], diff_out["noise"], mask=mask
        )
        total_loss += diff_loss_noise
        train_losses["diff_loss_noise"] = diff_loss_noise
        train_losses["total_loss"] = total_loss

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 0.5
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        train_losses["learning_rate"] = f"{self.optimizer.param_groups[0]['lr']:.1e}"
        train_losses["batch_size"] = batch["speaker_id"].shape[0]

        return (train_losses["total_loss"], train_losses, None)

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()
        if isinstance(self.w2v, dict):
            for key in self.w2v.keys():
                self.w2v[key].eval()
        else:
            self.w2v.eval()

        epoch_sum_loss: float = 0.0  # total loss
        # Put the data to cuda device
        device = self.accelerator.device
        with device:
            torch.cuda.empty_cache()
        self.model = self.model.to(device)
        self.w2v = self.w2v.to(device)

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
            speech = batch["speech"].cpu().numpy()
            speech = speech[0]
            self.batch_count += 1
            self.step += 1
            if len(speech) >= 16000 * 25:
                continue
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, _ = self._train_step(batch)

            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += total_loss
                self.current_loss = total_loss
                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )
                if self.accelerator.is_main_process and self.batch_count % 10 == 0:
                    self.echo_log(train_losses, mode="Training")

                self.save_checkpoint()
        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, None

    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # Dump config file
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)

        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        # Stop when meeting max epoch or self.cfg.train.num_train_steps
        while (
            self.epoch < self.max_epoch and self.step < self.cfg.train.num_train_steps
        ):
            if self.accelerator.is_main_process:
                self.logger.info("\n")
                self.logger.info("-" * 32)
                self.logger.info("Epoch {}: ".format(self.epoch))
                self.logger.info("Start training...")

            train_total_loss, _ = self._train_epoch()

            self.epoch += 1
            self.accelerator.wait_for_everyone()
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_total_loss
                    ),
                )
            )
        self.accelerator.end_training()
        if self.accelerator.is_main_process:
            self.logger.info("Training finished...")

    def save_checkpoint(self):
        self.accelerator.wait_for_everyone()
        # Main process only
        if self.accelerator.is_main_process:
            if self.batch_count % self.save_checkpoint_stride[0] == 0:
                keep_last = self.keep_last[0]
                # Read all folders in self.checkpoint_dir
                all_ckpts = os.listdir(self.checkpoint_dir)
                # Exclude non-folders
                all_ckpts = [
                    ckpt
                    for ckpt in all_ckpts
                    if os.path.isdir(os.path.join(self.checkpoint_dir, ckpt))
                ]
                if len(all_ckpts) > keep_last:
                    # Keep only the last keep_last folders in self.checkpoint_dir, sorted by step "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
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
        self.accelerator.wait_for_everyone()

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
        self.logger.info(", ".join(message))
