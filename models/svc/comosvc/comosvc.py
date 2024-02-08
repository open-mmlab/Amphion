# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import copy
import numpy as np
import math
from tqdm.auto import tqdm

from utils.ssim import SSIM

from models.svc.transformer.conformer import Conformer, BaseModule
from models.svc.diffusion.diffusion_wrapper import DiffusionWrapper


class Consistency(nn.Module):
    def __init__(self, cfg, distill=False):
        super().__init__()
        self.cfg = cfg
        self.denoise_fn = DiffusionWrapper(self.cfg)
        self.cfg = cfg.model.comosvc
        self.teacher = not distill
        self.P_mean = self.cfg.P_mean
        self.P_std = self.cfg.P_std
        self.sigma_data = self.cfg.sigma_data
        self.sigma_min = self.cfg.sigma_min
        self.sigma_max = self.cfg.sigma_max
        self.rho = self.cfg.rho
        self.N = self.cfg.n_timesteps
        self.ssim_loss = SSIM()

        # Time step discretization
        step_indices = torch.arange(self.N)
        # karras boundaries formula
        t_steps = (
            self.sigma_min ** (1 / self.rho)
            + step_indices
            / (self.N - 1)
            * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
        ) ** self.rho
        self.t_steps = torch.cat(
            [torch.zeros_like(t_steps[:1]), self.round_sigma(t_steps)]
        )

    def init_consistency_training(self):
        self.denoise_fn_ema = copy.deepcopy(self.denoise_fn)
        self.denoise_fn_pretrained = copy.deepcopy(self.denoise_fn)

    def EDMPrecond(self, x, sigma, cond, denoise_fn):
        """
        karras diffusion reverse process

        Args:
            x: noisy mel-spectrogram [B x n_mel x L]
            sigma: noise level [B x 1 x 1]
            cond: output of conformer encoder [B x n_mel x L]
            denoise_fn: denoiser neural network e.g. DilatedCNN

        Returns:
            denoised mel-spectrogram [B x n_mel x L]
        """
        sigma = sigma.reshape(-1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2).sqrt()
        )
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        x_in = x_in.transpose(1, 2)
        x = x.transpose(1, 2)
        cond = cond.transpose(1, 2)
        c_noise = c_noise.squeeze()
        if c_noise.dim() == 0:
            c_noise = c_noise.unsqueeze(0)
        F_x = denoise_fn(x_in, c_noise, cond)
        D_x = c_skip * x + c_out * (F_x)
        D_x = D_x.transpose(1, 2)
        return D_x

    def EDMLoss(self, x_start, cond, mask):
        """
        compute loss for EDM model

        Args:
            x_start: ground truth mel-spectrogram [B x n_mel x L]
            cond: output of conformer encoder [B x n_mel x L]
            mask: mask of padded frames [B x n_mel x L]
        """
        rnd_normal = torch.randn([x_start.shape[0], 1, 1], device=x_start.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # follow Grad-TTS, start from Gaussian noise with mean cond and std I
        noise = (torch.randn_like(x_start) + cond) * sigma
        D_yn = self.EDMPrecond(x_start + noise, sigma, cond, self.denoise_fn)
        loss = weight * ((D_yn - x_start) ** 2)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def edm_sampler(
        self,
        latents,
        cond,
        nonpadding,
        num_steps=50,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """
        karras diffusion sampler

        Args:
            latents: noisy mel-spectrogram [B x n_mel x L]
            cond: output of conformer encoder [B x n_mel x L]
            nonpadding: mask of padded frames [B x n_mel x L]
            num_steps: number of steps for diffusion inference

        Returns:
            denoised mel-spectrogram [B x n_mel x L]
        """
        # Time step discretization.

        num_steps = num_steps + 1
        step_indices = torch.arange(num_steps, device=latents.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop.
        x_next = latents * t_steps[0]
        # wrap in tqdm for progress bar
        bar = tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])))
        for i, (t_cur, t_next) in bar:
            x_cur = x_next
            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            t = torch.zeros((x_cur.shape[0], 1, 1), device=x_cur.device)
            t[:, 0, 0] = t_hat
            t_hat = t
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)
            # Euler step.
            denoised = self.EDMPrecond(x_hat, t_hat, cond, self.denoise_fn)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # add Heun’s 2nd order method
            # if i < num_steps - 1:
            #     t = torch.zeros((x_cur.shape[0], 1, 1), device=x_cur.device)
            #     t[:, 0, 0] = t_next
            #     #t_next = t
            #     denoised = self.EDMPrecond(x_next, t, cond, self.denoise_fn, nonpadding)
            #     d_prime = (x_next - denoised) / t_next
            #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def CTLoss_D(self, y, cond, mask):
        """
        compute loss for consistency distillation

        Args:
            y: ground truth mel-spectrogram [B x n_mel x L]
            cond: output of conformer encoder [B x n_mel x L]
            mask: mask of padded frames [B x n_mel x L]
        """
        with torch.no_grad():
            mu = 0.95
            for p, ema_p in zip(
                self.denoise_fn.parameters(), self.denoise_fn_ema.parameters()
            ):
                ema_p.mul_(mu).add_(p, alpha=1 - mu)

        n = torch.randint(1, self.N, (y.shape[0],))
        z = torch.randn_like(y) + cond

        tn_1 = self.t_steps[n + 1].reshape(-1, 1, 1).to(y.device)
        f_theta = self.EDMPrecond(y + tn_1 * z, tn_1, cond, self.denoise_fn)

        with torch.no_grad():
            tn = self.t_steps[n].reshape(-1, 1, 1).to(y.device)

            # euler step
            x_hat = y + tn_1 * z
            denoised = self.EDMPrecond(x_hat, tn_1, cond, self.denoise_fn_pretrained)
            d_cur = (x_hat - denoised) / tn_1
            y_tn = x_hat + (tn - tn_1) * d_cur

            # Heun’s 2nd order method

            denoised2 = self.EDMPrecond(y_tn, tn, cond, self.denoise_fn_pretrained)
            d_prime = (y_tn - denoised2) / tn
            y_tn = x_hat + (tn - tn_1) * (0.5 * d_cur + 0.5 * d_prime)

            f_theta_ema = self.EDMPrecond(y_tn, tn, cond, self.denoise_fn_ema)

        loss = (f_theta - f_theta_ema.detach()) ** 2
        loss = torch.sum(loss * mask) / torch.sum(mask)

        # check nan
        if torch.any(torch.isnan(loss)):
            print("nan loss")
        if torch.any(torch.isnan(f_theta)):
            print("nan f_theta")
        if torch.any(torch.isnan(f_theta_ema)):
            print("nan f_theta_ema")

        return loss

    def get_t_steps(self, N):
        N = N + 1
        step_indices = torch.arange(N)
        t_steps = (
            self.sigma_min ** (1 / self.rho)
            + step_indices
            / (N - 1)
            * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
        ) ** self.rho

        return t_steps.flip(0)

    def CT_sampler(self, latents, cond, nonpadding, t_steps=1):
        """
        consistency distillation sampler

        Args:
            latents: noisy mel-spectrogram [B x n_mel x L]
            cond: output of conformer encoder [B x n_mel x L]
            nonpadding: mask of padded frames [B x n_mel x L]
            t_steps: number of steps for diffusion inference

        Returns:
            denoised mel-spectrogram [B x n_mel x L]
        """
        # one-step
        if t_steps == 1:
            t_steps = [80]
        # multi-step
        else:
            t_steps = self.get_t_steps(t_steps)

        t_steps = torch.as_tensor(t_steps).to(latents.device)
        latents = latents * t_steps[0]
        _t = torch.zeros((latents.shape[0], 1, 1), device=latents.device)
        _t[:, 0, 0] = t_steps[0]
        x = self.EDMPrecond(latents, _t, cond, self.denoise_fn_ema)

        for t in t_steps[1:-1]:
            z = torch.randn_like(x) + cond
            x_tn = x + (t**2 - self.sigma_min**2).sqrt() * z
            _t = torch.zeros((x.shape[0], 1, 1), device=x.device)
            _t[:, 0, 0] = t
            t = _t
            x = self.EDMPrecond(x_tn, t, cond, self.denoise_fn_ema)
        return x

    def forward(self, x, nonpadding, cond, t_steps=1, infer=False):
        """
        calculate loss or sample mel-spectrogram

        Args:
            x:
                training: ground truth mel-spectrogram [B x n_mel x L]
                inference: output of encoder [B x n_mel x L]
        """
        if self.teacher:  # teacher model -- karras diffusion
            if not infer:
                loss = self.EDMLoss(x, cond, nonpadding)
                return loss
            else:
                shape = (cond.shape[0], self.cfg.n_mel, cond.shape[2])
                x = torch.randn(shape, device=x.device) + cond
                x = self.edm_sampler(x, cond, nonpadding, t_steps)

            return x
        else:  # Consistency distillation
            if not infer:
                loss = self.CTLoss_D(x, cond, nonpadding)
                return loss

            else:
                shape = (cond.shape[0], self.cfg.n_mel, cond.shape[2])
                x = torch.randn(shape, device=x.device) + cond
                x = self.CT_sampler(x, cond, nonpadding, t_steps=1)

            return x


class ComoSVC(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg.model.comosvc.n_mel = self.cfg.preprocess.n_mel
        self.distill = self.cfg.model.comosvc.distill
        self.encoder = Conformer(self.cfg.model.comosvc)
        self.decoder = Consistency(self.cfg, distill=self.distill)
        self.ssim_loss = SSIM()

    @torch.no_grad()
    def forward(self, x_mask, x, n_timesteps, temperature=1.0):
        """
        Generates mel-spectrogram from pitch, content vector, energy. Returns:
            1. encoder outputs (from conformer)
            2. decoder outputs (from diffusion-based decoder)

        Args:
            x_mask : mask of padded frames in mel-spectrogram. [B x L x n_mel]
            x : output of encoder framework. [B x L x d_condition]
            n_timesteps : number of steps to use for reverse diffusion in decoder.
            temperature : controls variance of terminal distribution.
        """

        # Get encoder_outputs `mu_x`
        mu_x = self.encoder(x, x_mask)
        encoder_outputs = mu_x

        mu_x = mu_x.transpose(1, 2)
        x_mask = x_mask.transpose(1, 2)

        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(
            mu_x, x_mask, mu_x, t_steps=n_timesteps, infer=True
        )
        decoder_outputs = decoder_outputs.transpose(1, 2)
        return encoder_outputs, decoder_outputs

    def compute_loss(self, x_mask, x, mel, skip_diff=False):
        """
        Computes 2 losses:
            1. prior loss: loss between mel-spectrogram and encoder outputs. (l2 and ssim loss)
            2. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x_mask : mask of padded frames in mel-spectrogram. [B x L x n_mel]
            x : output of encoder framework. [B x L x d_condition]
            mel : ground truth mel-spectrogram. [B x L x n_mel]
        """

        mu_x = self.encoder(x, x_mask)
        # prior loss
        x_mask = x_mask.repeat(1, 1, mel.shape[-1])
        prior_loss = torch.sum(
            0.5 * ((mel - mu_x) ** 2 + math.log(2 * math.pi)) * x_mask
        )

        prior_loss = prior_loss / (torch.sum(x_mask) * self.cfg.model.comosvc.n_mel)
        # ssim loss
        ssim_loss = self.ssim_loss(mu_x, mel)
        ssim_loss = torch.sum(ssim_loss * x_mask) / torch.sum(x_mask)

        x_mask = x_mask.transpose(1, 2)
        mu_x = mu_x.transpose(1, 2)
        mel = mel.transpose(1, 2)
        if not self.distill and skip_diff:
            diff_loss = prior_loss.clone()
            diff_loss.fill_(0)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        else:
            mu_y = mu_x
            mask_y = x_mask

            diff_loss = self.decoder(mel, mask_y, mu_y, infer=False)

        return ssim_loss, prior_loss, diff_loss
