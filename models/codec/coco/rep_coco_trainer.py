# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import numpy as np
import math
import torch.nn.functional as F


from models.base.base_trainer import BaseTrainer
from models.codec.coco.coco_dataset import CocoDataset, CocoCollator
from models.codec.coco.rep_coco_model import CocoContentStyle, CocoContent, CocoStyle

import whisper
import torchvision


class RepCocoTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(RepCocoTrainer, self).__init__(args, cfg)

        self._build_input_model()

    def _build_model(self):
        self.coco_model_type = getattr(
            self.cfg.model.coco, "coco_type", "content_style"
        )
        if self.coco_model_type == "content_style":
            model = CocoContentStyle(cfg=self.cfg.model.coco)
        elif self.coco_model_type == "content":
            model = CocoContent(cfg=self.cfg.model.coco)
        elif self.coco_model_type == "style":
            model = CocoStyle(cfg=self.cfg.model.coco)
        else:
            raise ValueError(f"Unknown coco type: {self.coco_model_type}")
        return model

    def _build_dataset(self):
        return CocoDataset, CocoCollator

    def spec_augment(self, mel, height):
        """
        Args:
            mel: tensor (..., n_mels, frames)
            height: int 68-92 for default 80 mels
        """
        tgt = torchvision.transforms.functional.resize(mel, (height, mel.shape[-1]))
        if height >= mel.shape[-2]:
            return tgt[:, : mel.shape[-2], :]
        else:
            silence = tgt[:, -1:, :].repeat(1, mel.shape[-2] - height, 1)
            silence += torch.randn_like(silence) / 10
            return torch.cat((tgt, silence), 1)

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)  # Specify bf16
    def _extract_whisper_features(self, wavs, frame_lens, spec_perturb=True):
        """
        Args:
            wavs: (B, T) at 16khz. Note that the max duration should be 30s
            frame_lens: (B,)
        Returns:
            features: (B, T, D)
        """
        # wavs: (batch, max_len)
        wavs = whisper.pad_or_trim(wavs)
        # batch_mel: (batch, 80, 3000)
        batch_mel = whisper.log_mel_spectrogram(wavs, device=self.accelerator.device)

        if spec_perturb:
            height = random.randint(68, 92)
            batch_mel = self.spec_augment(batch_mel, height)

        with torch.no_grad():
            # (batch, 1500, 1024)
            features = self.whisper_model.embed_audio(batch_mel)

        max_len = int(frame_lens.max().item())
        mask = torch.arange(features.size(1), device=features.device).expand(
            len(frame_lens), -1
        ) < frame_lens.unsqueeze(1)
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))

        if features.shape[1] >= max_len:
            features = features[:, :max_len, :]
        else:
            padding_frames = max_len - features.shape[1]
            last_frame = features[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            features = torch.cat([features, padding], dim=1)

        if self.use_normed_whisper:
            features = (features - self.whisper_mean) / self.whisper_std

        return features

    def _build_input_model(self):
        ## Whisper ##
        self.whisper_model = whisper.load_model(
            "medium", self.accelerator.device
        )  # 1024 dim
        self.whisper_model.eval()

        if self.accelerator.mixed_precision == "bf16":
            self.whisper_model = self.whisper_model.to(torch.bfloat16)

        self.use_normed_whisper = getattr(
            self.cfg.model.coco, "use_normed_whisper", False
        )
        if self.use_normed_whisper:
            whisper_stats = torch.load(
                self.cfg.model.coco.whisper_stats_path,
                map_location=self.accelerator.device,
            )
            self.whisper_mean = whisper_stats["mean"]  # (1024,)
            self.whisper_std = whisper_stats["std"]  # (1024,)

    @torch.no_grad()
    def _extract_mel_features(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.cfg.preprocess.mel_mean) / math.sqrt(
            self.cfg.preprocess.mel_var
        )
        return mel_feature

    def _compute_l1_loss(self, pred, gt, x_mask):
        """
        pred, gt: [B, T, D]
        x_mask: [B, T]
        """
        loss = F.l1_loss(pred, gt, reduction="none") * x_mask.unsqueeze(-1)  # [B, T, D]
        loss = torch.mean(loss, dim=2).sum() / x_mask.sum()
        return loss

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        speech = batch["wav"]  # [B, T_wav]
        speech_lens = batch["wav_len"]  # [B]
        x_mask = batch["mask"]  # [B, T]
        chromagram_feats = batch["chromagram"]  # [B, T, 24]

        if self.coco_model_type in ["content_style", "content"]:
            frame_lens = torch.sum(x_mask, dim=1).int()  # [B]
            whisper_feats = self._extract_whisper_features(
                batch["wav_16000"], frame_lens, spec_perturb=True
            )  # [B, T, D]

        torch.cuda.empty_cache()

        # TODO: write as a hyparam, check the weights
        rec_loss_weight = 32

        if self.coco_model_type == "content_style":
            whisper_rec, chromagram_rec, codebook_loss, _ = self.model(
                whisper_feats, chromagram_feats
            )

            whisper_rec_loss = self._compute_l1_loss(whisper_rec, whisper_feats, x_mask)
            chromagram_rec_loss = self._compute_l1_loss(
                chromagram_rec, chromagram_feats, x_mask
            )
            total_loss += (
                whisper_rec_loss * rec_loss_weight
                + chromagram_rec_loss * rec_loss_weight
            )
            train_losses["whisper_rec_loss"] = whisper_rec_loss
            train_losses["chromagram_rec_loss"] = chromagram_rec_loss

        elif self.coco_model_type == "content":
            whisper_rec, codebook_loss, _ = self.model(whisper_feats)

            whisper_rec_loss = self._compute_l1_loss(whisper_rec, whisper_feats, x_mask)
            total_loss += whisper_rec_loss * rec_loss_weight
            train_losses["whisper_rec_loss"] = whisper_rec_loss

        elif self.coco_model_type == "style":
            chromagram_rec, codebook_loss, _ = self.model(chromagram_feats)

            chromagram_rec_loss = self._compute_l1_loss(
                chromagram_rec, chromagram_feats, x_mask
            )
            total_loss += chromagram_rec_loss * rec_loss_weight
            train_losses["chromagram_rec_loss"] = chromagram_rec_loss

        total_loss += codebook_loss
        train_losses["codebook_loss"] = codebook_loss

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()

        train_losses["batch_size"] = speech.shape[0]
        train_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return (total_loss.item(), train_losses, train_stats)
