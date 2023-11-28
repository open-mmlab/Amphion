# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from models.base.base_trainer import BaseTrainer
from models.tta.autoencoder.autoencoder_dataset import (
    AutoencoderKLDataset,
    AutoencoderKLCollator,
)
from models.tta.autoencoder.autoencoder import AutoencoderKL
from models.tta.autoencoder.autoencoder_loss import AutoencoderLossWithDiscriminator
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader


class AutoencoderKLTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        BaseTrainer.__init__(self, args, cfg)
        self.cfg = cfg
        self.save_config_file()

    def build_dataset(self):
        return AutoencoderKLDataset, AutoencoderKLCollator

    def build_optimizer(self):
        opt_ae = torch.optim.AdamW(self.model.parameters(), **self.cfg.train.adam)
        opt_disc = torch.optim.AdamW(
            self.criterion.discriminator.parameters(), **self.cfg.train.adam
        )
        optimizer = {"opt_ae": opt_ae, "opt_disc": opt_disc}
        return optimizer

    def build_data_loader(self):
        Dataset, Collator = self.build_dataset()
        # build dataset instance for each dataset and combine them by ConcatDataset
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=False)
            datasets_list.append(subdataset)
        train_dataset = ConcatDataset(datasets_list)

        train_collate = Collator(self.cfg)

        # use batch_sampler argument instead of (sampler, shuffle, drop_last, batch_size)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_collate,
            num_workers=self.args.num_workers,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
        )
        if not self.cfg.train.ddp or self.args.local_rank == 0:
            datasets_list = []
            for dataset in self.cfg.dataset:
                subdataset = Dataset(self.cfg, dataset, is_valid=True)
                datasets_list.append(subdataset)
            valid_dataset = ConcatDataset(datasets_list)
            valid_collate = Collator(self.cfg)

            valid_loader = DataLoader(
                valid_dataset,
                collate_fn=valid_collate,
                num_workers=1,
                batch_size=self.cfg.train.batch_size,
            )
        else:
            raise NotImplementedError("DDP is not supported yet.")
            # valid_loader = None
        data_loader = {"train": train_loader, "valid": valid_loader}
        return data_loader

    # TODO: check it...
    def build_scheduler(self):
        return None
        # return ReduceLROnPlateau(self.optimizer["opt_ae"], **self.cfg.train.lronPlateau)

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def build_criterion(self):
        return AutoencoderLossWithDiscriminator(self.cfg.model.loss)

    def get_state_dict(self):
        if self.scheduler != None:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer_ae": self.optimizer["opt_ae"].state_dict(),
                "optimizer_disc": self.optimizer["opt_disc"].state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "batch_size": self.cfg.train.batch_size,
            }
        else:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer_ae": self.optimizer["opt_ae"].state_dict(),
                "optimizer_disc": self.optimizer["opt_disc"].state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "batch_size": self.cfg.train.batch_size,
            }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer["opt_ae"].load_state_dict(checkpoint["optimizer_ae"])
        self.optimizer["opt_disc"].load_state_dict(checkpoint["optimizer_disc"])
        if self.scheduler != None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def build_model(self):
        self.model = AutoencoderKL(self.cfg.model.autoencoderkl)
        return self.model

    # TODO: train step
    def train_step(self, data):
        global_step = self.step
        optimizer_idx = global_step % 2

        train_losses = {}
        total_loss = 0
        train_states = {}

        inputs = data["melspec"].unsqueeze(1)  # (B, 80, T) -> (B, 1, 80, T)
        reconstructions, posterior = self.model(inputs)
        # train_stats.update(stat)

        train_losses = self.criterion(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posterior,
            optimizer_idx=optimizer_idx,
            global_step=global_step,
            last_layer=self.model.get_last_layer(),
            split="train",
        )

        if optimizer_idx == 0:
            total_loss = train_losses["loss"]
            self.optimizer["opt_ae"].zero_grad()
            total_loss.backward()
            self.optimizer["opt_ae"].step()

        else:
            total_loss = train_losses["d_loss"]
            self.optimizer["opt_disc"].zero_grad()
            total_loss.backward()
            self.optimizer["opt_disc"].step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        return train_losses, train_states, total_loss.item()

    # TODO: eval step
    @torch.no_grad()
    def eval_step(self, data, index):
        valid_loss = {}
        total_valid_loss = 0
        valid_stats = {}

        inputs = data["melspec"].unsqueeze(1)  # (B, 80, T) -> (B, 1, 80, T)
        reconstructions, posterior = self.model(inputs)

        loss = F.l1_loss(inputs, reconstructions)
        valid_loss["loss"] = loss

        total_valid_loss += loss

        for item in valid_loss:
            valid_loss[item] = valid_loss[item].item()

        return valid_loss, valid_stats, total_valid_loss.item()
