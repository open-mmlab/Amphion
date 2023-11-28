# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = x + residual
        return x


class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        kernel_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(
            in_dims, compute_dims, kernel_size=kernel_size, bias=False
        )
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    def __init__(
        self, feat_dims, upsample_scales, compute_dims, res_blocks, res_out_dims, pad
    ):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            kernel_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / kernel_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent : -self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class WaveRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.pad = self.cfg.VOCODER.MEL_FRAME_PAD

        if self.cfg.VOCODER.MODE == "mu_law_quantize":
            self.n_classes = 2**self.cfg.VOCODER.BITS
        elif self.cfg.VOCODER.MODE == "mu_law" or self.cfg.VOCODER:
            self.n_classes = 30

        self._to_flatten = []

        self.rnn_dims = self.cfg.VOCODER.RNN_DIMS
        self.aux_dims = self.cfg.VOCODER.RES_OUT_DIMS // 4
        self.hop_length = self.cfg.VOCODER.HOP_LENGTH
        self.fc_dims = self.cfg.VOCODER.FC_DIMS
        self.upsample_factors = self.cfg.VOCODER.UPSAMPLE_FACTORS
        self.feat_dims = self.cfg.VOCODER.INPUT_DIM
        self.compute_dims = self.cfg.VOCODER.COMPUTE_DIMS
        self.res_out_dims = self.cfg.VOCODER.RES_OUT_DIMS
        self.res_blocks = self.cfg.VOCODER.RES_BLOCKS

        self.upsample = UpsampleNetwork(
            self.feat_dims,
            self.upsample_factors,
            self.compute_dims,
            self.res_blocks,
            self.res_out_dims,
            self.pad,
        )
        self.I = nn.Linear(self.feat_dims + self.aux_dims + 1, self.rnn_dims)

        self.rnn1 = nn.GRU(self.rnn_dims, self.rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(
            self.rnn_dims + self.aux_dims, self.rnn_dims, batch_first=True
        )
        self._to_flatten += [self.rnn1, self.rnn2]

        self.fc1 = nn.Linear(self.rnn_dims + self.aux_dims, self.fc_dims)
        self.fc2 = nn.Linear(self.fc_dims + self.aux_dims, self.fc_dims)
        self.fc3 = nn.Linear(self.fc_dims, self.n_classes)

        self.num_params()

        self._flatten_parameters()

    def forward(self, x, mels):
        device = next(self.parameters()).device

        self._flatten_parameters()

        batch_size = x.size(0)
        h1 = torch.zeros(1, batch_size, self.rnn_dims, device=device)
        h2 = torch.zeros(1, batch_size, self.rnn_dims, device=device)
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0] : aux_idx[1]]
        a2 = aux[:, :, aux_idx[1] : aux_idx[2]]
        a3 = aux[:, :, aux_idx[2] : aux_idx[3]]
        a4 = aux[:, :, aux_idx[3] : aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters

    def _flatten_parameters(self):
        [m.flatten_parameters() for m in self._to_flatten]
