# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch import nn

LRELU_SLOPE = 0.1


# This code is a refined MRD adopted from BigVGAN under the MIT License
# https://github.com/NVIDIA/BigVGAN


class DiscriminatorR(nn.Module):
    def __init__(self, cfg, resolution):
        super().__init__()

        self.resolution = resolution
        assert (
            len(self.resolution) == 3
        ), "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = LRELU_SLOPE

        norm_f = (
            weight_norm if cfg.model.mrd.use_spectral_norm == False else spectral_norm
        )
        if cfg.model.mrd.mrd_override:
            print(
                "INFO: overriding MRD use_spectral_norm as {}".format(
                    cfg.model.mrd.mrd_use_spectral_norm
                )
            )
            norm_f = (
                weight_norm
                if cfg.model.mrd.mrd_use_spectral_norm == False
                else spectral_norm
            )
        self.d_mult = cfg.model.mrd.discriminator_channel_mult_factor
        if cfg.model.mrd.mrd_override:
            print(
                "INFO: overriding mrd channel multiplier as {}".format(
                    cfg.model.mrd.mrd_channel_mult
                )
            )
            self.d_mult = cfg.model.mrd.mrd_channel_mult

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.model.mrd.resolutions
        assert (
            len(self.resolutions) == 3
        ), "MRD requires list of list with len=3, each element having a list with len=3. got {}".format(
            self.resolutions
        )
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
