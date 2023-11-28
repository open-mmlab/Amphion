# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.svc.base import SVCInference
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.comosvc.comosvc import ComoSVC


class ComoSVCInference(SVCInference):
    def __init__(self, args, cfg, infer_type="from_dataset"):
        SVCInference.__init__(self, args, cfg, infer_type)

    def _build_model(self):
        # TODO: sort out the config
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = ComoSVC(self.cfg)
        if self.cfg.model.comosvc.distill:
            self.acoustic_mapper.decoder.init_consistency_training()
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        return model

    def _inference_each_batch(self, batch_data):
        device = self.accelerator.device
        for k, v in batch_data.items():
            batch_data[k] = v.to(device)

        cond = self.condition_encoder(batch_data)
        mask = batch_data["mask"]
        encoder_pred, decoder_pred = self.acoustic_mapper(
            mask, cond, self.cfg.inference.comosvc.inference_steps
        )

        return decoder_pred
