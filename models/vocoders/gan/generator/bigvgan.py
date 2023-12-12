# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import torch.nn as nn

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from modules.vocoder_blocks import *
from modules.activation_functions import *
from modules.anti_aliasing import *

LRELU_SLOPE = 0.1

# The AMPBlock Module is adopted from BigVGAN under the MIT License
# https://github.com/NVIDIA/BigVGAN


class AMPBlock1(torch.nn.Module):
    def __init__(
        self, cfg, channels, kernel_size=3, dilation=(1, 3, 5), activation=None
    ):
        super(AMPBlock1, self).__init__()
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

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # total number of conv layers

        if (
            activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(
                            channels, alpha_logscale=cfg.model.bigvgan.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(
                            channels, alpha_logscale=cfg.model.bigvgan.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
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

        self.num_layers = len(self.convs)  # total number of conv layers

        if (
            activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(
                            channels, alpha_logscale=cfg.model.bigvgan.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(
                            channels, alpha_logscale=cfg.model.bigvgan.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(torch.nn.Module):
    def __init__(self, cfg):
        super(BigVGAN, self).__init__()
        self.cfg = cfg

        self.num_kernels = len(cfg.model.bigvgan.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.model.bigvgan.upsample_rates)

        # Conv pre to boost channels
        self.conv_pre = weight_norm(
            Conv1d(
                cfg.preprocess.n_mel,
                cfg.model.bigvgan.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )

        resblock = AMPBlock1 if cfg.model.bigvgan.resblock == "1" else AMPBlock2

        # Upsamplers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                cfg.model.bigvgan.upsample_rates,
                cfg.model.bigvgan.upsample_kernel_sizes,
            )
        ):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                cfg.model.bigvgan.upsample_initial_channel // (2**i),
                                cfg.model.bigvgan.upsample_initial_channel
                                // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Res Blocks with AMP and Anti-aliasing
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg.model.bigvgan.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    cfg.model.bigvgan.resblock_kernel_sizes,
                    cfg.model.bigvgan.resblock_dilation_sizes,
                )
            ):
                self.resblocks.append(
                    resblock(cfg, ch, k, d, activation=cfg.model.bigvgan.activation)
                )

        # Conv post for result
        if cfg.model.bigvgan.activation == "snake":
            activation_post = Snake(ch, alpha_logscale=cfg.model.bigvgan.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif cfg.model.bigvgan.activation == "snakebeta":
            activation_post = SnakeBeta(
                ch, alpha_logscale=cfg.model.bigvgan.snake_logscale
            )
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # Weight Norm
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
