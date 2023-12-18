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
from utils.util import Logger, ValueWindow
from torch.utils.data import ConcatDataset, DataLoader
from models.tts.base.tts_trainer import TTSTrainer
from models.base.base_trainer import BaseTrainer
from models.base.base_sampler import VariableSampler
from models.tts.naturalspeech2.ns2_dataset import NS2Dataset, NS2Collator, batch_by_size
from models.tts.naturalspeech2.ns2_loss import (
    log_pitch_loss,
    log_dur_loss,
    diff_loss,
    diff_ce_loss,
)
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from models.tts.naturalspeech2.ns2 import NaturalSpeech2
from torch.optim import Adam, AdamW
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from diffusers import get_scheduler

import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration


class NS2Trainer(TTSTrainer):
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

        self.time_window = ValueWindow(50)

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
                    f"Model parameters: {self._count_parameters(self.model)/1e6:.2f}M"
                )

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
            (
                self.train_dataloader,
                self.valid_dataloader,
            ) = self.accelerator.prepare(
                self.train_dataloader,
                self.valid_dataloader,
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
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # TODO: Resume from ckpt need test/debug
        with self.accelerator.main_process_first():
            if args.resume:
                if self.accelerator.is_main_process:
                    self.logger.info("Resuming from checkpoint...")
                start = time.monotonic_ns()
                ckpt_path = self._load_model(
                    self.checkpoint_dir,
                    args.checkpoint_path,
                    resume_type=args.resume_type,
                )
                end = time.monotonic_ns()
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )
                self.checkpoints_path = json.load(
                    open(os.path.join(ckpt_path, "ckpts.json"), "r")
                )

            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.accelerator.is_main_process:
                self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        # Only for TTS tasks
        self.task_type = "TTS"
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
        model = NaturalSpeech2(cfg=self.cfg.model)
        return model

    def _build_dataset(self):
        return NS2Dataset, NS2Collator

    def _build_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize......")
            Dataset, Collator = self._build_dataset()
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
            np.random.seed(980205)
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
            )
            self.accelerator.wait_for_everyone()

            valid_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=True)
            valid_collate = Collator(self.cfg)
            batch_sampler = batch_by_size(
                valid_dataset.num_frame_indices,
                valid_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]
            valid_loader = DataLoader(
                valid_dataset,
                collate_fn=valid_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(batches, drop_last=False),
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

        else:
            print("Use Normal Batchsize......")
            Dataset, Collator = self._build_dataset()
            train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=train_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            valid_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=True)
            valid_collate = Collator(self.cfg)

            valid_loader = DataLoader(
                valid_dataset,
                shuffle=True,
                collate_fn=valid_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

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

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        code = batch["code"]  # (B, 16, T)
        pitch = batch["pitch"]  # (B, T)
        duration = batch["duration"]  # (B, N)
        phone_id = batch["phone_id"]  # (B, N)
        ref_code = batch["ref_code"]  # (B, 16, T')
        phone_mask = batch["phone_mask"]  # (B, N)
        mask = batch["mask"]  # (B, T)
        ref_mask = batch["ref_mask"]  # (B, T')

        diff_out, prior_out = self.model(
            code=code,
            pitch=pitch,
            duration=duration,
            phone_id=phone_id,
            ref_code=ref_code,
            phone_mask=phone_mask,
            mask=mask,
            ref_mask=ref_mask,
        )

        # pitch loss
        pitch_loss = log_pitch_loss(prior_out["pitch_pred_log"], pitch, mask=mask)
        total_loss += pitch_loss
        train_losses["pitch_loss"] = pitch_loss

        # duration loss
        dur_loss = log_dur_loss(prior_out["dur_pred_log"], duration, mask=phone_mask)
        total_loss += dur_loss
        train_losses["dur_loss"] = dur_loss

        x0 = self.model.module.code_to_latent(code)
        if self.cfg.model.diffusion.diffusion_type == "diffusion":
            # diff loss x0
            diff_loss_x0 = diff_loss(diff_out["x0_pred"], x0, mask=mask)
            total_loss += diff_loss_x0
            train_losses["diff_loss_x0"] = diff_loss_x0

            # diff loss noise
            diff_loss_noise = diff_loss(
                diff_out["noise_pred"], diff_out["noise"], mask=mask
            )
            total_loss += diff_loss_noise * self.cfg.train.diff_noise_loss_lambda
            train_losses["diff_loss_noise"] = diff_loss_noise

        elif self.cfg.model.diffusion.diffusion_type == "flow":
            # diff flow matching loss
            flow_gt = diff_out["noise"] - x0
            diff_loss_flow = diff_loss(diff_out["flow_pred"], flow_gt, mask=mask)
            total_loss += diff_loss_flow
            train_losses["diff_loss_flow"] = diff_loss_flow

        # diff loss ce

        # (nq, B, T); (nq, B, T, 1024)
        if self.cfg.train.diff_ce_loss_lambda > 0:
            pred_indices, pred_dist = self.model.module.latent_to_code(
                diff_out["x0_pred"], nq=code.shape[1]
            )
            gt_indices, _ = self.model.module.latent_to_code(x0, nq=code.shape[1])
            diff_loss_ce = diff_ce_loss(pred_dist, gt_indices, mask=mask)
            total_loss += diff_loss_ce * self.cfg.train.diff_ce_loss_lambda
            train_losses["diff_loss_ce"] = diff_loss_ce

        self.optimizer.zero_grad()
        # total_loss.backward()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 0.5
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        if self.cfg.train.diff_ce_loss_lambda > 0:
            pred_indices_list = pred_indices.long().detach().cpu().numpy()
            gt_indices_list = gt_indices.long().detach().cpu().numpy()
            mask_list = batch["mask"].detach().cpu().numpy()

            for i in range(pred_indices_list.shape[0]):
                pred_acc = np.sum(
                    (pred_indices_list[i] == gt_indices_list[i]) * mask_list
                ) / np.sum(mask_list)
                train_losses["pred_acc_{}".format(str(i))] = pred_acc

        train_losses["batch_size"] = code.shape[0]
        train_losses["max_frame_nums"] = np.max(
            batch["frame_nums"].detach().cpu().numpy()
        )

        return (total_loss.item(), train_losses, train_stats)

    @torch.inference_mode()
    def _valid_step(self, batch):
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        code = batch["code"]  # (B, 16, T)
        pitch = batch["pitch"]  # (B, T)
        duration = batch["duration"]  # (B, N)
        phone_id = batch["phone_id"]  # (B, N)
        ref_code = batch["ref_code"]  # (B, 16, T')
        phone_mask = batch["phone_mask"]  # (B, N)
        mask = batch["mask"]  # (B, T)
        ref_mask = batch["ref_mask"]  # (B, T')

        diff_out, prior_out = self.model(
            code=code,
            pitch=pitch,
            duration=duration,
            phone_id=phone_id,
            ref_code=ref_code,
            phone_mask=phone_mask,
            mask=mask,
            ref_mask=ref_mask,
        )

        # pitch loss
        pitch_loss = log_pitch_loss(prior_out["pitch_pred_log"], pitch, mask=mask)
        total_loss += pitch_loss
        valid_losses["pitch_loss"] = pitch_loss

        # duration loss
        dur_loss = log_dur_loss(prior_out["dur_pred_log"], duration, mask=phone_mask)
        total_loss += dur_loss
        valid_losses["dur_loss"] = dur_loss

        x0 = self.model.module.code_to_latent(code)
        if self.cfg.model.diffusion.diffusion_type == "diffusion":
            # diff loss x0
            diff_loss_x0 = diff_loss(diff_out["x0_pred"], x0, mask=mask)
            total_loss += diff_loss_x0
            valid_losses["diff_loss_x0"] = diff_loss_x0

            # diff loss noise
            diff_loss_noise = diff_loss(
                diff_out["noise_pred"], diff_out["noise"], mask=mask
            )
            total_loss += diff_loss_noise * self.cfg.train.diff_noise_loss_lambda
            valid_losses["diff_loss_noise"] = diff_loss_noise

        elif self.cfg.model.diffusion.diffusion_type == "flow":
            # diff flow matching loss
            flow_gt = diff_out["noise"] - x0
            diff_loss_flow = diff_loss(diff_out["flow_pred"], flow_gt, mask=mask)
            total_loss += diff_loss_flow
            valid_losses["diff_loss_flow"] = diff_loss_flow

        # diff loss ce

        # (nq, B, T); (nq, B, T, 1024)
        if self.cfg.train.diff_ce_loss_lambda > 0:
            pred_indices, pred_dist = self.model.module.latent_to_code(
                diff_out["x0_pred"], nq=code.shape[1]
            )
            gt_indices, _ = self.model.module.latent_to_code(x0, nq=code.shape[1])
            diff_loss_ce = diff_ce_loss(pred_dist, gt_indices, mask=mask)
            total_loss += diff_loss_ce * self.cfg.train.diff_ce_loss_lambda
            valid_losses["diff_loss_ce"] = diff_loss_ce

        for item in valid_losses:
            valid_losses[item] = valid_losses[item].item()

        if self.cfg.train.diff_ce_loss_lambda > 0:
            pred_indices_list = pred_indices.long().detach().cpu().numpy()
            gt_indices_list = gt_indices.long().detach().cpu().numpy()
            mask_list = batch["mask"].detach().cpu().numpy()

            for i in range(pred_indices_list.shape[0]):
                pred_acc = np.sum(
                    (pred_indices_list[i] == gt_indices_list[i]) * mask_list
                ) / np.sum(mask_list)
                valid_losses["pred_acc_{}".format(str(i))] = pred_acc

        return (total_loss.item(), valid_losses, valid_stats)

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = dict()

        for batch in self.valid_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss = total_loss
            for key, value in valid_losses.items():
                epoch_losses[key] = value

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

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
                    % (1 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)

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

            valid_total_loss, valid_losses = self._valid_epoch()
            if isinstance(valid_losses, dict):
                for key, loss in valid_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

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

            # Check if hit save_checkpoint_stride and run_eval
            run_eval = False
            if self.accelerator.is_main_process:
                save_checkpoint = False
                hit_dix = []
                for i, num in enumerate(self.save_checkpoint_stride):
                    if self.epoch % num == 0:
                        save_checkpoint = True
                        hit_dix.append(i)
                        run_eval |= self.run_eval[i]

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and save_checkpoint:
                path = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_total_loss
                    ),
                )
                print("save state......")
                self.accelerator.save_state(path)
                print("finish saving state......")
                json.dump(
                    self.checkpoints_path,
                    open(os.path.join(path, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
                # Remove old checkpoints
                to_remove = []
                for idx in hit_dix:
                    self.checkpoints_path[idx].append(path)
                    while len(self.checkpoints_path[idx]) > self.keep_last[idx]:
                        to_remove.append((idx, self.checkpoints_path[idx].pop(0)))

                # Search conflicts
                total = set()
                for i in self.checkpoints_path:
                    total |= set(i)
                do_remove = set()
                for idx, path in to_remove[::-1]:
                    if path in total:
                        self.checkpoints_path[idx].insert(0, path)
                    else:
                        do_remove.add(path)

                # Remove old checkpoints
                for path in do_remove:
                    shutil.rmtree(path, ignore_errors=True)
                    if self.accelerator.is_main_process:
                        self.logger.debug(f"Remove old checkpoint: {path}")

            self.accelerator.wait_for_everyone()
            if run_eval:
                # TODO: run evaluation
                pass

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
