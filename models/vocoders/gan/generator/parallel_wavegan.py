# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import numpy as np
import fsspec
import torch
from torch.nn import functional as F
from torch.nn.utils.parametrize import remove_parametrizations


class ResidualBlock(torch.nn.Module):
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
        self.conv = torch.nn.Conv1d(res_channels, gate_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.conv1x1_aux = torch.nn.Conv1d(aux_channels, gate_channels, 1, bias=False) if aux_channels > 0 else None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = torch.nn.Conv1d(gate_out_channels, res_channels, 1, bias=bias)
        self.conv1x1_skip = torch.nn.Conv1d(gate_out_channels, skip_channels, 1, bias=bias)

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


class Stretch2d(torch.nn.Module):
    def __init__(self, x_scale, y_scale, mode="nearest"):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class UpsampleNetwork(torch.nn.Module):
    def __init__(
        self,
        upsample_factors,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode="nearest",
        freq_axis_kernel_size=1,
        use_causal_conv=False,
    ):
        super().__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList([
            Stretch2d(scale, 1, interpolate_mode) for scale in upsample_factors
        ] + [
            torch.nn.Conv2d(1, 1, kernel_size=(freq_axis_kernel_size, scale * 2 + 1), padding=((freq_axis_kernel_size - 1) // 2, scale), bias=False)
            for scale in upsample_factors
        ] + [
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params) if nonlinear_activation is not None else torch.nn.Identity()
            for _ in upsample_factors
        ])

    def forward(self, c):
        c = c.unsqueeze(1)
        for f in self.up_layers:
            c = f(c)
        return c.squeeze(1)


class ConvUpsample(torch.nn.Module):
    def __init__(
        self,
        upsample_factors,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode="nearest",
        freq_axis_kernel_size=1,
        aux_channels=80,
        aux_context_window=0,
        use_causal_conv=False,
    ):
        super().__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        self.conv_in = torch.nn.Conv1d(aux_channels, aux_channels, kernel_size=kernel_size, bias=False)
        self.upsample = UpsampleNetwork(
            upsample_factors=upsample_factors,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
        )

    def forward(self, c):
        c_ = self.conv_in(c)
        c = c_[:, :, : -self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


def load_fsspec(path, map_location=None, cache=True, **kwargs):
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(f"filecache::{path}", mode="rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)


class ParallelWaveganGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        num_res_blocks=30,
        stacks=3,
        res_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        dropout=0.0,
        bias=True,
        use_weight_norm=True,
        upsample_factors=[4, 4, 4, 4],
        inference_padding=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.num_res_blocks = num_res_blocks
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.upsample_factors = upsample_factors
        self.upsample_scale = np.prod(upsample_factors)
        self.inference_padding = inference_padding
        self.use_weight_norm = use_weight_norm

        assert num_res_blocks % stacks == 0
        layers_per_stack = num_res_blocks // stacks

        self.first_conv = torch.nn.Conv1d(in_channels, res_channels, kernel_size=1, bias=True)
        self.upsample_net = ConvUpsample(upsample_factors=upsample_factors)
        self.conv_layers = torch.nn.ModuleList([
            ResidualBlock(
                kernel_size=kernel_size,
                res_channels=res_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=2 ** (layer % layers_per_stack),
                dropout=dropout,
                bias=bias,
            )
            for layer in range(num_res_blocks)
        ])
        self.last_conv_layers = torch.nn.ModuleList([
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(skip_channels, skip_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(skip_channels, out_channels, kernel_size=1, bias=True),
        ])

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, c):
        x = torch.randn([c.shape[0], 1, c.shape[2] * self.upsample_scale])
        x = x.to(self.first_conv.bias.device)
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.shape[-1] == x.shape[-1], f"Upsampling scale does not match the expected output. {c.shape} vs {x.shape}"
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        return x

    @torch.no_grad()
    def inference(self, c):
        c = c.to(self.first_conv.weight.device)
        c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                remove_parametrizations(m, "weight")
            except ValueError:
                pass
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2**x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

    def load_checkpoint(self, config, checkpoint_path, eval=False, cache=False):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training
            if self.use_weight_norm:
                self.remove_weight_norm()
