# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#################### NSF ####################

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# This code is adopted from Xin Wang's NSF under the MIT License
# https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts


class SineGen(nn.Module):
    def __init__(
        self, fs, harmonic_num=0, amp=0.1, noise_std=0.003, voiced_threshold=0
    ):
        super(SineGen, self).__init__()
        self.amp = amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.fs = fs
        self.voice_threshold = voiced_threshold

    def _f0toUnvoiced(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voice_threshold)
        return uv

    @torch.no_grad()
    def forward(self, f0, upp):
        f0 = f0.unsqueeze(-1)
        fn = torch.multiply(
            f0, torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1)
        )
        rad_values = (fn / self.fs) % 1
        rand_ini = torch.rand(fn.shape[0], fn.shape[2], device=fn.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        is_half = rad_values.dtype is not torch.float32
        tmp_over_one = torch.cumsum(rad_values.double(), 1)
        if is_half:
            tmp_over_one = tmp_over_one.half()
        else:
            tmp_over_one = tmp_over_one.float()
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1),
            scale_factor=upp,
            mode="linear",
            align_corners=True,
        ).transpose(2, 1)
        rad_values = F.interpolate(
            rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
        ).transpose(2, 1)
        tmp_over_one %= 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * (-1.0)
        rad_values = rad_values.double()
        cumsum_shift = cumsum_shift.double()
        sine_waves = torch.sin(
            torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
        )
        if is_half:
            sine_waves = sine_waves.half()
        else:
            sine_waves = sine_waves.float()
        sine_waves = sine_waves * self.amp
        uv = self._f0toUnvoiced(f0)
        uv = F.interpolate(
            uv.transpose(2, 1), scale_factor=upp, mode="nearest"
        ).transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
