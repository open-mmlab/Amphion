# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from tqdm import tqdm
from models.tts.base import TTSTrainer
from models.tts.fastspeech2.fs2 import FastSpeech2, FastSpeech2Loss
from models.tts.fastspeech2.fs2_dataset import FS2Dataset, FS2Collator
from optimizer.optimizers import NoamLR


class FastSpeech2Trainer(TTSTrainer):
    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)
        self.cfg = cfg

    def _build_dataset(self):
        return FS2Dataset, FS2Collator

    def __build_scheduler(self):
        return NoamLR(self.optimizer, **self.cfg.train.lr_scheduler)

    def _write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar("train/" + key, value, self.step)
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.sw.add_scalar("learning_rate", lr, self.step)

    def _write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar("val/" + key, value, self.step)

    def _build_criterion(self):
        return FastSpeech2Loss(self.cfg)

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

    def _build_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **self.cfg.train.adam)
        return optimizer

    def _build_scheduler(self):
        scheduler = NoamLR(self.optimizer, **self.cfg.train.lr_scheduler)
        return scheduler

    def _build_model(self):
        self.model = FastSpeech2(self.cfg)
        return self.model

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.train()
        epoch_sum_loss: float = 0.0
        epoch_step: int = 0
        epoch_losses: dict = {}
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
            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                loss, train_losses = self._train_step(batch)
                self.accelerator.backward(loss)
                grad_clip_thresh = self.cfg.train.grad_clip_thresh
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_thresh)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            self.batch_count += 1

            # Update info for each step
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += loss
                for key, value in train_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

                self.accelerator.log(
                    {
                        "Step/Train Loss": loss,
                        "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.step,
                )
                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()

        epoch_sum_loss = (
            epoch_sum_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )

        for key in epoch_losses.keys():
            epoch_losses[key] = (
                epoch_losses[key]
                / len(self.train_dataloader)
                * self.cfg.train.gradient_accumulation_step
            )
        return epoch_sum_loss, epoch_losses

    def _train_step(self, data):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        preds = self.model(data)

        train_losses = self.criterion(data, preds)

        total_loss = train_losses["loss"]
        for key, value in train_losses.items():
            train_losses[key] = value.item()

        return total_loss, train_losses

    @torch.no_grad()
    def _valid_step(self, data):
        valid_loss = {}
        total_valid_loss = 0
        valid_stats = {}

        preds = self.model(data)

        valid_losses = self.criterion(data, preds)

        total_valid_loss = valid_losses["loss"]
        for key, value in valid_losses.items():
            valid_losses[key] = value.item()

        return total_valid_loss, valid_losses, valid_stats
