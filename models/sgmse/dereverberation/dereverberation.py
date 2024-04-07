import torch
from torch import nn
from types import SimpleNamespace
from modules.sgmse.sdes import SDERegistry
from modules.sgmse.shared import BackboneRegistry
import json


class ScoreModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(cfg.backbone)
        dnn_cfg = cfg[cfg.backbone]
        self.dnn = dnn_cls(**dnn_cfg)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(cfg.sde)
        sde_cfg = cfg[cfg.sde]
        self.sde = sde_cls(**sde_cfg)

    def forward(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        score = -self.dnn(dnn_input, t)
        return score
