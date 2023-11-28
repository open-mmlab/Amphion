# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import nn
from modules.vocoder_blocks import *

from einops import rearrange
import torchaudio.transforms as T

from nnAudio import features

LRELU_SLOPE = 0.1


class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg, hop_length, n_octaves, bins_per_octave):
        super(DiscriminatorCQT, self).__init__()
        self.cfg = cfg

        self.filters = cfg.model.mssbcqtd.filters
        self.max_filters = cfg.model.mssbcqtd.max_filters
        self.filters_scale = cfg.model.mssbcqtd.filters_scale
        self.kernel_size = (3, 9)
        self.dilations = cfg.model.mssbcqtd.dilations
        self.stride = (1, 2)

        self.in_channels = cfg.model.mssbcqtd.in_channels
        self.out_channels = cfg.model.mssbcqtd.out_channels
        self.fs = cfg.preprocess.sample_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                NormConv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            NormConv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(self.kernel_size, (dilation, 1)),
                    norm="weight_norm",
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
                norm="weight_norm",
            )
        )

        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            norm="weight_norm",
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=LRELU_SLOPE)
        self.resample = T.Resample(orig_freq=self.fs, new_freq=self.fs * 2)

    def forward(self, x):
        fmap = []

        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = rearrange(z, "b c w t -> b c t w")

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleSubbandCQTDiscriminator, self).__init__()

        self.cfg = cfg

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    cfg,
                    hop_length=cfg.model.mssbcqtd.hop_lengths[i],
                    n_octaves=cfg.model.mssbcqtd.n_octaves[i],
                    bins_per_octave=cfg.model.mssbcqtd.bins_per_octaves[i],
                )
                for i in range(len(cfg.model.mssbcqtd.hop_lengths))
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
