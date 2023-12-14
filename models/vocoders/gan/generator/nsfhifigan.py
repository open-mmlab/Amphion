# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from modules.neural_source_filter import *
from modules.vocoder_blocks import *


LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.cfg = cfg

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock1, self).__init__()
        self.cfg = cfg

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


# This NSF Module is adopted from Xin Wang's NSF under the MIT License
# https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self, fs, harmonic_num=0, amp=0.1, noise_std=0.003, voiced_threshold=0
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.amp = amp
        self.noise_std = noise_std
        self.l_sin_gen = SineGen(fs, harmonic_num, amp, noise_std, voiced_threshold)

        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upp):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class NSFHiFiGAN(nn.Module):
    def __init__(self, cfg):
        super(NSFHiFiGAN, self).__init__()

        self.cfg = cfg
        self.num_kernels = len(self.cfg.model.nsfhifigan.resblock_kernel_sizes)
        self.num_upsamples = len(self.cfg.model.nsfhifigan.upsample_rates)
        self.m_source = SourceModuleHnNSF(
            fs=self.cfg.preprocess.sample_rate,
            harmonic_num=self.cfg.model.nsfhifigan.harmonic_num,
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(
            Conv1d(
                self.cfg.preprocess.n_mel,
                self.cfg.model.nsfhifigan.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )

        resblock = ResBlock1 if self.cfg.model.nsfhifigan.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                self.cfg.model.nsfhifigan.upsample_rates,
                self.cfg.model.nsfhifigan.upsample_kernel_sizes,
            )
        ):
            c_cur = self.cfg.model.nsfhifigan.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        self.cfg.model.nsfhifigan.upsample_initial_channel // (2**i),
                        self.cfg.model.nsfhifigan.upsample_initial_channel
                        // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(self.cfg.model.nsfhifigan.upsample_rates):
                stride_f0 = int(
                    np.prod(self.cfg.model.nsfhifigan.upsample_rates[i + 1 :])
                )
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        ch = self.cfg.model.nsfhifigan.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(
                zip(
                    self.cfg.model.nsfhifigan.resblock_kernel_sizes,
                    self.cfg.model.nsfhifigan.resblock_dilation_sizes,
                )
            ):
                self.resblocks.append(resblock(cfg, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(self.cfg.model.nsfhifigan.upsample_rates))

    def forward(self, x, f0):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)

            length = min(x.shape[-1], x_source.shape[-1])
            x = x[:, :, :length]
            x_source = x[:, :, :length]

            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
