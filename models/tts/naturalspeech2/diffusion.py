# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.tts.naturalspeech2.wavenet import WaveNet


class Diffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

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

        cum_beta = self.get_cum_beta(diffusion_step.unsqueeze(-1).unsqueeze(-1))
        x0_pred = self.diff_estimator(xt, x_mask, cond, diffusion_step, spk_query_emb)
        mean_pred = x0_pred * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        variance = (self.sigma**2) * (1.0 - torch.exp(-cum_beta / (self.sigma**2)))
        noise_pred = (xt - mean_pred) / (torch.sqrt(variance) * self.noise_factor)
        noise = z
        diff_out = {"x0_pred": x0_pred, "noise_pred": noise_pred, "noise": noise}
        return diff_out

    @torch.no_grad()
    def get_cum_beta(self, time_step):
        return self.beta_min * time_step + 0.5 * (self.beta_max - self.beta_min) * (
            time_step**2
        )

    @torch.no_grad()
    def get_beta_t(self, time_step):
        return self.beta_min + (self.beta_max - self.beta_min) * time_step

    @torch.no_grad()
    def forward_diffusion(self, x0, diffusion_step):
        """
        x0: (B, 128, T)
        time_step: (B,)
        """
        time_step = diffusion_step.unsqueeze(-1).unsqueeze(-1)
        cum_beta = self.get_cum_beta(time_step)
        mean = x0 * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        variance = (self.sigma**2) * (1 - torch.exp(-cum_beta / (self.sigma**2)))
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean + z * torch.sqrt(variance) * self.noise_factor
        return xt, z

    @torch.no_grad()
    def cal_dxt(self, xt, x_mask, cond, spk_query_emb, diffusion_step, h):
        time_step = diffusion_step.unsqueeze(-1).unsqueeze(-1)
        cum_beta = self.get_cum_beta(time_step=time_step)
        beta_t = self.get_beta_t(time_step=time_step)
        x0_pred = self.diff_estimator(xt, x_mask, cond, diffusion_step, spk_query_emb)
        mean_pred = x0_pred * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        noise_pred = xt - mean_pred
        variance = (self.sigma**2) * (1.0 - torch.exp(-cum_beta / (self.sigma**2)))
        logp = -noise_pred / (variance + 1e-8)
        dxt = -0.5 * h * beta_t * (logp + xt / (self.sigma**2))
        return dxt

    @torch.no_grad()
    def reverse_diffusion(self, z, x_mask, cond, n_timesteps, spk_query_emb):
        h = 1.0 / max(n_timesteps, 1)
        xt = z
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            dxt = self.cal_dxt(xt, x_mask, cond, spk_query_emb, diffusion_step=t, h=h)
            xt_ = xt - dxt
            if self.cfg.ode_solver == "midpoint":
                x_mid = 0.5 * (xt_ + xt)
                dxt = self.cal_dxt(
                    x_mid, x_mask, cond, spk_query_emb, diffusion_step=t + 0.5 * h, h=h
                )
                xt = xt - dxt
            elif self.cfg.ode_solver == "euler":
                xt = xt_
        return xt

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self, z, x_mask, cond, n_timesteps, spk_query_emb, t_start
    ):
        h = t_start / max(n_timesteps, 1)
        xt = z
        for i in range(n_timesteps):
            t = (t_start - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            dxt = self.cal_dxt(xt, x_mask, cond, spk_query_emb, diffusion_step=t, h=h)
            xt_ = xt - dxt
            if self.cfg.ode_solver == "midpoint":
                x_mid = 0.5 * (xt_ + xt)
                dxt = self.cal_dxt(
                    x_mid, x_mask, cond, spk_query_emb, diffusion_step=t + 0.5 * h, h=h
                )
                xt = xt - dxt
            elif self.cfg.ode_solver == "euler":
                xt = xt_
        return xt
