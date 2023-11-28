# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.svc.base import SVCTrainer
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.transformer.transformer import Transformer
from models.svc.transformer.conformer import Conformer
from utils.ssim import SSIM


class TransformerTrainer(SVCTrainer):
    def __init__(self, args, cfg):
        SVCTrainer.__init__(self, args, cfg)
        self.ssim_loss = SSIM()

    def _build_model(self):
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        if self.cfg.model.transformer.type == "transformer":
            self.acoustic_mapper = Transformer(self.cfg.model.transformer)
        elif self.cfg.model.transformer.type == "conformer":
            self.acoustic_mapper = Conformer(self.cfg.model.transformer)
        else:
            raise NotImplementedError
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        return model

    def _forward_step(self, batch):
        total_loss = 0
        device = self.accelerator.device
        mel = batch["mel"]
        mask = batch["mask"]

        condition = self.condition_encoder(batch)
        mel_pred = self.acoustic_mapper(condition, mask)

        l1_loss = torch.sum(torch.abs(mel_pred - mel) * batch["mask"]) / torch.sum(
            batch["mask"]
        )
        self._check_nan(l1_loss, mel_pred, mel)
        total_loss += l1_loss
        ssim_loss = self.ssim_loss(mel_pred, mel)
        ssim_loss = torch.sum(ssim_loss * batch["mask"]) / torch.sum(batch["mask"])
        self._check_nan(ssim_loss, mel_pred, mel)
        total_loss += ssim_loss

        return total_loss
