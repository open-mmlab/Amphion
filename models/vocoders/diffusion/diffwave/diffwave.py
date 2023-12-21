# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This model code is adopted from DiffWave/model.py under the Apache License
# https://github.com/lmnt-com/diffwave
# Only the config-related varaible names are changed.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, upsample_factors):
        super().__init__()
        self.conv1 = ConvTranspose2d(
            1,
            1,
            [3, upsample_factors[0] * 2],
            stride=[1, upsample_factors[0]],
            padding=[1, upsample_factors[0] // 2],
        )
        self.conv2 = ConvTranspose2d(
            1,
            1,
            [3, upsample_factors[1] * 2],
            stride=[1, upsample_factors[1]],
            padding=[1, upsample_factors[1] // 2],
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(512, residual_channels)

        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step

        conditioner = self.conditioner_projection(conditioner)
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg.model.diffwave.noise_schedule = np.linspace(
            self.cfg.model.diffwave.noise_schedule_factors[0],
            self.cfg.model.diffwave.noise_schedule_factors[1],
            self.cfg.model.diffwave.noise_schedule_factors[2],
        ).tolist()
        self.input_projection = Conv1d(1, self.cfg.model.diffwave.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(
            len(self.cfg.model.diffwave.noise_schedule)
        )
        self.spectrogram_upsampler = SpectrogramUpsampler(
            self.cfg.model.diffwave.upsample_factors
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    self.cfg.preprocess.n_mel,
                    self.cfg.model.diffwave.residual_channels,
                    2 ** (i % self.cfg.model.diffwave.dilation_cycle_length),
                )
                for i in range(self.cfg.model.diffwave.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            self.cfg.model.diffwave.residual_channels,
            self.cfg.model.diffwave.residual_channels,
            1,
        )
        self.output_projection = Conv1d(self.cfg.model.diffwave.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
