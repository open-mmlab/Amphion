# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from models.tts.naturalspeech2.base_trainer import TTSTrainer
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


class NS2Trainer(TTSTrainer):
    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)

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
            **self.cfg.train.adam
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

    # def _train_epoch(self):
    #     ...
