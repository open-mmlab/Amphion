# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.base.base_trainer import BaseTrainer
from diffusers import DDPMScheduler
from models.tta.ldm.audioldm_dataset import AudioLDMDataset, AudioLDMCollator
from models.tta.autoencoder.autoencoder import AutoencoderKL
from models.tta.ldm.audioldm import AudioLDM, UNetModel
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from transformers import T5EncoderModel
from diffusers import DDPMScheduler


class AudioLDMTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        BaseTrainer.__init__(self, args, cfg)
        self.cfg = cfg

        self.build_autoencoderkl()
        self.build_textencoder()
        self.nosie_scheduler = self.build_noise_scheduler()

        self.save_config_file()

    def build_autoencoderkl(self):
        self.autoencoderkl = AutoencoderKL(self.cfg.model.autoencoderkl)
        self.autoencoder_path = self.cfg.model.autoencoder_path
        checkpoint = torch.load(self.autoencoder_path, map_location="cpu")
        self.autoencoderkl.load_state_dict(checkpoint["model"])
        self.autoencoderkl.cuda(self.args.local_rank)
        self.autoencoderkl.requires_grad_(requires_grad=False)
        self.autoencoderkl.eval()

    def build_textencoder(self):
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.text_encoder.cuda(self.args.local_rank)
        self.text_encoder.requires_grad_(requires_grad=False)
        self.text_encoder.eval()

    def build_noise_scheduler(self):
        nosie_scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg.model.noise_scheduler.num_train_timesteps,
            beta_start=self.cfg.model.noise_scheduler.beta_start,
            beta_end=self.cfg.model.noise_scheduler.beta_end,
            beta_schedule=self.cfg.model.noise_scheduler.beta_schedule,
            clip_sample=self.cfg.model.noise_scheduler.clip_sample,
            # steps_offset=self.cfg.model.noise_scheduler.steps_offset,
            # set_alpha_to_one=self.cfg.model.noise_scheduler.set_alpha_to_one,
            # skip_prk_steps=self.cfg.model.noise_scheduler.skip_prk_steps,
            prediction_type=self.cfg.model.noise_scheduler.prediction_type,
        )
        return nosie_scheduler

    def build_dataset(self):
        return AudioLDMDataset, AudioLDMCollator

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

    def build_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfg.train.adam)
        return optimizer

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
        criterion = nn.MSELoss(reduction="mean")
        return criterion

    def get_state_dict(self):
        if self.scheduler != None:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "batch_size": self.cfg.train.batch_size,
            }
        else:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
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
        if self.scheduler != None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def build_model(self):
        self.model = AudioLDM(self.cfg.model.audioldm)
        return self.model

    @torch.no_grad()
    def mel_to_latent(self, melspec):
        posterior = self.autoencoderkl.encode(melspec)
        latent = posterior.sample()  # (B, 4, 5, 78)
        return latent

    @torch.no_grad()
    def get_text_embedding(self, text_input_ids, text_attention_mask):
        text_embedding = self.text_encoder(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state
        return text_embedding  # (B, T, 768)

    def train_step(self, data):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        melspec = data["melspec"].unsqueeze(1)  # (B, 80, T) -> (B, 1, 80, T)
        latents = self.mel_to_latent(melspec)

        text_embedding = self.get_text_embedding(
            data["text_input_ids"], data["text_attention_mask"]
        )

        noise = torch.randn_like(latents).float()

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.cfg.model.noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        with torch.no_grad():
            noisy_latents = self.nosie_scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.model(
            noisy_latents, timesteps=timesteps, context=text_embedding
        )

        loss = self.criterion(model_pred, noise)

        train_losses["loss"] = loss
        total_loss += loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        return train_losses, train_stats, total_loss.item()

    # TODO: eval step
    @torch.no_grad()
    def eval_step(self, data, index):
        valid_loss = {}
        total_valid_loss = 0
        valid_stats = {}

        melspec = data["melspec"].unsqueeze(1)  # (B, 80, T) -> (B, 1, 80, T)
        latents = self.mel_to_latent(melspec)

        text_embedding = self.get_text_embedding(
            data["text_input_ids"], data["text_attention_mask"]
        )

        noise = torch.randn_like(latents).float()

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.cfg.model.noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        noisy_latents = self.nosie_scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.model(noisy_latents, timesteps, text_embedding)

        loss = self.criterion(model_pred, noise)
        valid_loss["loss"] = loss

        total_valid_loss += loss

        for item in valid_loss:
            valid_loss[item] = valid_loss[item].item()

        return valid_loss, valid_stats, total_valid_loss.item()
