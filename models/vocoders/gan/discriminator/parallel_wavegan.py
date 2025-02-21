# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import remove_parametrizations


class ResidualBlock(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        res_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        dropout=0.0,
        dilation=1,
        bias=True,
        use_causal_conv=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_causal_conv = use_causal_conv
        padding = (kernel_size - 1) * dilation if use_causal_conv else ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(res_channels, gate_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.conv1x1_aux = nn.Conv1d(aux_channels, gate_channels, 1, bias=False) if aux_channels > 0 else None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = nn.Conv1d(gate_out_channels, res_channels, 1, bias=bias)
        self.conv1x1_skip = nn.Conv1d(gate_out_channels, skip_channels, 1, bias=bias)

    def forward(self, x, c):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)
        x = x[:, :, :residual.size(-1)] if self.use_causal_conv else x
        xa, xb = x.split(x.size(1) // 2, dim=1)
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(1) // 2, dim=1)
            xa, xb = xa + ca, xb + cb
        x = torch.tanh(xa) * torch.sigmoid(xb)
        s = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * (0.5**2)
        return x, s


class ParallelWaveganDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        num_layers=10,
        conv_channels=64,
        dilation_factor=1,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        bias=True,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, " [!] does not support even number kernel size."
        assert dilation_factor > 0, " [!] dilation factor must be > 0."
        self.conv_layers = nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(num_layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor**i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                nn.Conv1d(
                    conv_in_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params),
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer = nn.Conv1d(conv_in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [last_conv_layer]
        self.apply_weight_norm()

    def forward(self, x):
        for f in self.conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                print(f"Weight norm is removed from {m}.")
                remove_parametrizations(m, "weight")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class ResidualParallelWaveganDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        num_layers=30,
        stacks=3,
        res_channels=64,
        gate_channels=128,
        skip_channels=64,
        dropout=0.0,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.res_factor = math.sqrt(1.0 / num_layers)

        assert num_layers % stacks == 0
        layers_per_stack = num_layers // stacks

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, res_channels, kernel_size=1, padding=0, dilation=1, bias=True),
            getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params),
        )

        self.conv_layers = nn.ModuleList()
        for layer in range(num_layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                res_channels=res_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=-1,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=False,
            )
            self.conv_layers += [conv]

        self.last_conv_layers = nn.ModuleList(
            [
                getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params),
                nn.Conv1d(skip_channels, skip_channels, kernel_size=1, padding=0, dilation=1, bias=True),
                getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params),
                nn.Conv1d(skip_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=True),
            ]
        )

        self.apply_weight_norm()

    def forward(self, x):
        x = self.first_conv(x)

        skips = 0
        for f in self.conv_layers:
            x, h = f(x, None)
            skips += h
        skips *= self.res_factor

        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                print(f"Weight norm is removed from {m}.")
                remove_parametrizations(m, "weight")
            except ValueError: 
                return

        self.apply(_remove_weight_norm)
