# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.tts.naturalspeech2.wavenet import WaveNet


class DiffusionFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.diff_estimator = WaveNet(cfg.wavenet)
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max
        self.sigma = cfg.sigma
        self.noise_factor = cfg.noise_factor

    def forward(self, x, x_mask, cond, spk_query_emb, offset=1e-5):
        """
        x: (B, 128, T)
        x_mask: (B, T), mask is 0
        cond: (B, T, 512)
        spk_query_emb: (B, 32, 512)
        """
        diffusion_step = torch.rand(
            x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False
        )
        diffusion_step = torch.clamp(diffusion_step, offset, 1.0 - offset)
        xt, z = self.forward_diffusion(x0=x, diffusion_step=diffusion_step)

        flow_pred = self.diff_estimator(
            xt, x_mask, cond, diffusion_step, spk_query_emb
        )  # noise - x0_pred, noise_pred - x0
        noise = z
        x0_pred = noise - flow_pred
        noise_pred = x + flow_pred
        diff_out = {
            "x0_pred": x0_pred,
            "noise_pred": noise_pred,
            "noise": noise,
            "flow_pred": flow_pred,
        }
        return diff_out

    @torch.no_grad()
    def forward_diffusion(self, x0, diffusion_step):
        """
        x0: (B, 128, T)
        time_step: (B,)
        """
        time_step = diffusion_step.unsqueeze(-1).unsqueeze(-1)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = (1 - time_step) * x0 + time_step * z
        return xt, z

    @torch.no_grad()
    def cal_dxt(self, xt, x_mask, cond, spk_query_emb, diffusion_step, h):
        flow_pred = self.diff_estimator(
            xt, x_mask, cond, diffusion_step, spk_query_emb
        )  # z - x0 = x1 - x0
        dxt = h * flow_pred
        return dxt

    @torch.no_grad()
    def reverse_diffusion(self, z, x_mask, cond, n_timesteps, spk_query_emb):
        h = 1.0 / n_timesteps
        xt = z
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            dxt = self.cal_dxt(xt, x_mask, cond, spk_query_emb, diffusion_step=t, h=h)
            xt = xt - dxt
        return xt
