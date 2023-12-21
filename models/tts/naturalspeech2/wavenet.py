# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()

        self.gain = Linear(cond_dim, in_dim)
        self.bias = Linear(cond_dim, in_dim)

        nn.init.xavier_uniform_(self.gain.weight)
        nn.init.constant_(self.gain.bias, 1)

        nn.init.xavier_uniform_(self.bias.weight)
        nn.init.constant_(self.bias.bias, 0)

    def forward(self, x, condition):
        gain = self.gain(condition)
        bias = self.bias(condition)
        if gain.dim() == 2:
            gain = gain.unsqueeze(-1)
        if bias.dim() == 2:
            bias = bias.unsqueeze(-1)
        return x * gain + bias


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Linear(*args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    layer.weight.data.normal_(0.0, 0.02)
    return layer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, attn_head, dilation, drop_out, has_cattn=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dilation = dilation
        self.has_cattn = has_cattn
        self.attn_head = attn_head
        self.drop_out = drop_out

        self.dilated_conv = Conv1d(
            hidden_dim, 2 * hidden_dim, 3, padding=dilation, dilation=dilation
        )
        self.diffusion_proj = Linear(hidden_dim, hidden_dim)

        self.cond_proj = Conv1d(hidden_dim, hidden_dim * 2, 1)
        self.out_proj = Conv1d(hidden_dim, hidden_dim * 2, 1)

        if self.has_cattn:
            self.attn = nn.MultiheadAttention(
                hidden_dim, attn_head, 0.1, batch_first=True
            )
            self.film = FiLM(hidden_dim * 2, hidden_dim)

            self.ln = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(self.drop_out)

    def forward(self, x, x_mask, cond, diffusion_step, spk_query_emb):
        diffusion_step = self.diffusion_proj(diffusion_step).unsqueeze(-1)  # (B, d, 1)
        cond = self.cond_proj(cond)  # (B, 2*d, T)

        y = x + diffusion_step
        if x_mask != None:
            y = y * x_mask.to(y.dtype)[:, None, :]  # (B, 2*d, T)

        if self.has_cattn:
            y_ = y.transpose(1, 2)
            y_ = self.ln(y_)

            y_, _ = self.attn(y_, spk_query_emb, spk_query_emb)  # (B, T, d)

        y = self.dilated_conv(y) + cond  # (B, 2*d, T)

        if self.has_cattn:
            y = self.film(y.transpose(1, 2), y_)  # (B, T, 2*d)
            y = y.transpose(1, 2)  # (B, 2*d, T)

        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)

        y = self.out_proj(y)

        residual, skip = torch.chunk(y, 2, dim=1)

        if x_mask != None:
            residual = residual * x_mask.to(y.dtype)[:, None, :]
            skip = skip * x_mask.to(y.dtype)[:, None, :]

        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.in_dim = cfg.input_size
        self.hidden_dim = cfg.hidden_size
        self.out_dim = cfg.out_size
        self.num_layers = cfg.num_layers
        self.cross_attn_per_layer = cfg.cross_attn_per_layer
        self.dilation_cycle = cfg.dilation_cycle
        self.attn_head = cfg.attn_head
        self.drop_out = cfg.drop_out

        self.in_proj = Conv1d(self.in_dim, self.hidden_dim, 1)
        self.diffusion_embedding = SinusoidalPosEmb(self.hidden_dim)

        self.mlp = nn.Sequential(
            Linear(self.hidden_dim, self.hidden_dim * 4),
            Mish(),
            Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        self.cond_ln = nn.LayerNorm(self.hidden_dim)

        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    self.hidden_dim,
                    self.attn_head,
                    2 ** (i % self.dilation_cycle),
                    self.drop_out,
                    has_cattn=(i % self.cross_attn_per_layer == 0),
                )
                for i in range(self.num_layers)
            ]
        )

        self.skip_proj = Conv1d(self.hidden_dim, self.hidden_dim, 1)
        self.out_proj = Conv1d(self.hidden_dim, self.out_dim, 1)

        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x, x_mask, cond, diffusion_step, spk_query_emb):
        """
        x: (B, 128, T)
        x_mask: (B, T), mask is 0
        cond: (B, T, 512)
        diffusion_step: (B,)
        spk_query_emb: (B, 32, 512)
        """
        cond = self.cond_ln(cond)
        cond_input = cond.transpose(1, 2)

        x_input = self.in_proj(x)

        x_input = F.relu(x_input)

        diffusion_step = self.diffusion_embedding(diffusion_step).to(x.dtype)
        diffusion_step = self.mlp(diffusion_step)

        skip = []
        for _, layer in enumerate(self.layers):
            x_input, skip_connection = layer(
                x_input, x_mask, cond_input, diffusion_step, spk_query_emb
            )
            skip.append(skip_connection)

        x_input = torch.sum(torch.stack(skip), dim=0) / math.sqrt(self.num_layers)

        x_out = self.skip_proj(x_input)

        x_out = F.relu(x_out)

        x_out = self.out_proj(x_out)  # (B, 128, T)

        return x_out
