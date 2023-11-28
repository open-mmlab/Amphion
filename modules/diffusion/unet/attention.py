# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.general.utils import Conv1d, normalization, zero_module
from .basic import UNetBlock


class AttentionBlock(UNetBlock):
    r"""A spatial transformer encoder block that allows spatial positions to attend
    to each other. Reference from `latent diffusion repo
    <https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/attention.py#L531>`_.

    Args:
        channels: Number of channels in the input.
        num_head_channels: Number of channels per attention head.
        num_heads: Number of attention heads. Overrides ``num_head_channels`` if set.
        encoder_channels: Number of channels in the encoder output for cross-attention.
            If ``None``, then self-attention is performed.
        use_self_attention: Whether to use self-attention before cross-attention, only applicable if encoder_channels is set.
        dims: Number of spatial dimensions, i.e. 1 for temporal signals, 2 for images.
        h_dim: The dimension of the height, would be applied if ``dims`` is 2.
        encoder_hdim: The dimension of the height of the encoder output, would be applied if ``dims`` is 2.
        p_dropout: Dropout probability.
    """

    def __init__(
        self,
        channels: int,
        num_head_channels: int = 32,
        num_heads: int = -1,
        encoder_channels: int = None,
        use_self_attention: bool = False,
        dims: int = 1,
        h_dim: int = 100,
        encoder_hdim: int = 384,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.p_dropout = p_dropout
        self.dims = dims

        if dims == 1:
            self.channels = channels
        elif dims == 2:
            # We consider the channel as product of channel and height, i.e. C x H
            # This is because we want to apply attention on the audio signal, which is 1D
            self.channels = channels * h_dim
        else:
            raise ValueError(f"invalid number of dimensions: {dims}")

        if num_head_channels == -1:
            assert (
                self.channels % num_heads == 0
            ), f"q,k,v channels {self.channels} is not divisible by num_heads {num_heads}"
            self.num_heads = num_heads
            self.num_head_channels = self.channels // num_heads
        else:
            assert (
                self.channels % num_head_channels == 0
            ), f"q,k,v channels {self.channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = self.channels // num_head_channels
            self.num_head_channels = num_head_channels

        if encoder_channels is not None:
            self.use_self_attention = use_self_attention

            if dims == 1:
                self.encoder_channels = encoder_channels
            elif dims == 2:
                self.encoder_channels = encoder_channels * encoder_hdim
            else:
                raise ValueError(f"invalid number of dimensions: {dims}")

            if use_self_attention:
                self.self_attention = BasicAttentionBlock(
                    self.channels,
                    self.num_head_channels,
                    self.num_heads,
                    p_dropout=self.p_dropout,
                )
            self.cross_attention = BasicAttentionBlock(
                self.channels,
                self.num_head_channels,
                self.num_heads,
                self.encoder_channels,
                p_dropout=self.p_dropout,
            )
        else:
            self.encoder_channels = None
            self.self_attention = BasicAttentionBlock(
                self.channels,
                self.num_head_channels,
                self.num_heads,
                p_dropout=self.p_dropout,
            )

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor = None):
        r"""
        Args:
            x: input tensor with shape [B x ``channels`` x ...]
            encoder_output: feature tensor with shape [B x ``encoder_channels`` x ...], if ``None``, then self-attention is performed.

        Returns:
            output tensor with shape [B x ``channels`` x ...]
        """
        shape = x.size()
        x = x.reshape(shape[0], self.channels, -1).contiguous()

        if self.encoder_channels is None:
            assert (
                encoder_output is None
            ), "encoder_output must be None for self-attention."
            h = self.self_attention(x)

        else:
            assert (
                encoder_output is not None
            ), "encoder_output must be given for cross-attention."
            encoder_output = encoder_output.reshape(
                shape[0], self.encoder_channels, -1
            ).contiguous()

            if self.use_self_attention:
                x = self.self_attention(x)
            h = self.cross_attention(x, encoder_output)

        return h.reshape(*shape).contiguous()


class BasicAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_head_channels: int = 32,
        num_heads: int = -1,
        context_channels: int = None,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.p_dropout = p_dropout
        self.context_channels = context_channels

        if num_head_channels == -1:
            assert (
                self.channels % num_heads == 0
            ), f"q,k,v channels {self.channels} is not divisible by num_heads {num_heads}"
            self.num_heads = num_heads
            self.num_head_channels = self.channels // num_heads
        else:
            assert (
                self.channels % num_head_channels == 0
            ), f"q,k,v channels {self.channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = self.channels // num_head_channels
            self.num_head_channels = num_head_channels

        if context_channels is not None:
            self.to_q = nn.Sequential(
                normalization(self.channels),
                Conv1d(self.channels, self.channels, 1),
            )
            self.to_kv = Conv1d(context_channels, 2 * self.channels, 1)
        else:
            self.to_qkv = nn.Sequential(
                normalization(self.channels),
                Conv1d(self.channels, 3 * self.channels, 1),
            )

        self.linear = Conv1d(self.channels, self.channels)

        self.proj_out = nn.Sequential(
            normalization(self.channels),
            Conv1d(self.channels, self.channels, 1),
            nn.GELU(),
            nn.Dropout(p=self.p_dropout),
            zero_module(Conv1d(self.channels, self.channels, 1)),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor = None):
        r"""
        Args:
            q: input tensor with shape [B, ``channels``, L]
            kv: feature tensor with shape [B, ``context_channels``, T], if ``None``, then self-attention is performed.

        Returns:
            output tensor with shape [B, ``channels``, L]
        """
        N, C, L = q.size()

        if self.context_channels is not None:
            assert kv is not None, "kv must be given for cross-attention."

            q = (
                self.to_q(q)
                .reshape(self.num_heads, self.num_head_channels, -1)
                .transpose(-1, -2)
                .contiguous()
            )
            kv = (
                self.to_kv(kv)
                .reshape(2, self.num_heads, self.num_head_channels, -1)
                .transpose(-1, -2)
                .chunk(2)
            )
            k, v = (
                kv[0].squeeze(0).contiguous(),
                kv[1].squeeze(0).contiguous(),
            )

        else:
            qkv = (
                self.to_qkv(q)
                .reshape(3, self.num_heads, self.num_head_channels, -1)
                .transpose(-1, -2)
                .chunk(3)
            )
            q, k, v = (
                qkv[0].squeeze(0).contiguous(),
                qkv[1].squeeze(0).contiguous(),
                qkv[2].squeeze(0).contiguous(),
            )

        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_dropout).transpose(
            -1, -2
        )
        h = h.reshape(N, -1, L).contiguous()
        h = self.linear(h)

        x = q + h
        h = self.proj_out(x)

        return x + h
