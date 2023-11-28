# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict

from models.svc.base import SVCInference
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.transformer.transformer import Transformer
from models.svc.transformer.conformer import Conformer


class TransformerInference(SVCInference):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        SVCInference.__init__(self, args, cfg, infer_type)

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

    def _inference_each_batch(self, batch_data):
        device = self.accelerator.device
        for k, v in batch_data.items():
            batch_data[k] = v.to(device)

        condition = self.condition_encoder(batch_data)
        y_pred = self.acoustic_mapper(condition, batch_data["mask"])

        return y_pred
