# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn as nn

from modules.general.utils import Conv1d, zero_module
from .residual_block import ResidualBlock


class BiDilConv(nn.Module):
    r"""Dilated CNN architecture with residual connections, default diffusion decoder.

    Args:
        input_channel: The number of input channels.
        base_channel: The number of base channels.
        n_res_block: The number of residual blocks.
        conv_kernel_size: The kernel size of convolutional layers.
        dilation_cycle_length: The cycle length of dilation.
        conditioner_size: The size of conditioner.
    """

    def __init__(
        self,
        input_channel,
        base_channel,
        n_res_block,
        conv_kernel_size,
        dilation_cycle_length,
        conditioner_size,
        output_channel: int = -1,
    ):
        super().__init__()

        self.input_channel = input_channel
        self.base_channel = base_channel
        self.n_res_block = n_res_block
        self.conv_kernel_size = conv_kernel_size
        self.dilation_cycle_length = dilation_cycle_length
        self.conditioner_size = conditioner_size
        self.output_channel = output_channel if output_channel > 0 else input_channel

        self.input = nn.Sequential(
            Conv1d(
                input_channel,
                base_channel,
                1,
            ),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=base_channel,
                    kernel_size=conv_kernel_size,
                    dilation=2 ** (i % dilation_cycle_length),
                    d_context=conditioner_size,
                )
                for i in range(n_res_block)
            ]
        )

        self.out_proj = nn.Sequential(
            Conv1d(
                base_channel,
                base_channel,
                1,
            ),
            nn.ReLU(),
            zero_module(
                Conv1d(
                    base_channel,
                    self.output_channel,
                    1,
                ),
            ),
        )

    def forward(self, x, y, context=None):
        """
        Args:
            x: Noisy mel-spectrogram [B x ``n_mel`` x L]
            y: FILM embeddings with the shape of (B, ``base_channel``)
            context: Context with the shape of [B x ``d_context`` x L], default to None.
        """

        h = self.input(x)

        skip = None
        for i in range(self.n_res_block):
            h, skip_connection = self.residual_blocks[i](h, y, context)
            skip = skip_connection if skip is None else skip_connection + skip

        out = skip / math.sqrt(self.n_res_block)

        out = self.out_proj(out)

        return out
