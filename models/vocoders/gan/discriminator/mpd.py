# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d, Conv1d
from torch.nn.utils import weight_norm, spectral_norm
from torch import nn
from modules.vocoder_blocks import *
from models.vocoders.gan.discriminator.msd import MultiScaleDiscriminator_JETS

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(self, cfg, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.d_mult = cfg.model.mpd.discriminator_channel_mult_factor
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        int(32 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(32 * self.d_mult),
                        int(128 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(128 * self.d_mult),
                        int(512 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(512 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(1024 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(2, 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, cfg):
        super(MultiPeriodDiscriminator, self).__init__()
        self.mpd_reshapes = cfg.model.mpd.mpd_reshapes
        print("mpd_reshapes: {}".format(self.mpd_reshapes))
        discriminators = [
            DiscriminatorP(cfg, rs, use_spectral_norm=cfg.model.mpd.use_spectral_norm)
            for rs in self.mpd_reshapes
        ]
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# TODO: merge with DiscriminatorP (lmxue, yicheng)
class DiscriminatorP_vits(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP_vits, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# TODO: merge with MultiPeriodDiscriminator (lmxue, yicheng)
class MultiPeriodDiscriminator_vits(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator_vits, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP_vits(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        outputs = {
            "y_d_hat_r": y_d_rs,
            "y_d_hat_g": y_d_gs,
            "fmap_rs": fmap_rs,
            "fmap_gs": fmap_gs,
        }

        return outputs


class DiscriminatorP_JETS(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP_JETS, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)
        fmap.append(x)

        return x, fmap


class MultiPeriodDiscriminator_JETS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator_JETS, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [
            DiscriminatorP_JETS(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs


# JETS Multi-scale Multi-period discriminator module.
class MultiScaleMultiPeriodDiscriminator(torch.nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(self, use_spectral_norm=False):
        super(MultiScaleMultiPeriodDiscriminator, self).__init__()

        self.msd = MultiScaleDiscriminator_JETS()
        self.mpd = MultiPeriodDiscriminator_JETS()

    def forward(self, y):

        _, msd_outs_d_rs = self.msd(y)
        # msd_outs = self.msd(y, y_hat)
        _, mpd_outs_d_rs = self.mpd(y)
        # mpd_outs = self.mpd(y, y_hat)
        return msd_outs_d_rs + mpd_outs_d_rs
        # ground_truth, generated
        # return msd_outs + mpd_outs
