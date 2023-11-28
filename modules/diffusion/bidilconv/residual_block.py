# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn

from modules.activation_functions import GaU
from modules.general.utils import Conv1d


class ResidualBlock(nn.Module):
    r"""Residual block with dilated convolution, main portion of ``BiDilConv``.

    Args:
        channels: The number of channels of input and output.
        kernel_size: The kernel size of dilated convolution.
        dilation: The dilation rate of dilated convolution.
        d_context: The dimension of content encoder output, None if don't use context.
    """

    def __init__(
        self,
        channels: int = 256,
        kernel_size: int = 3,
        dilation: int = 1,
        d_context: int = None,
    ):
        super().__init__()

        self.context = d_context

        self.gau = GaU(
            channels,
            kernel_size,
            dilation,
            d_context,
        )

        self.out_proj = Conv1d(
            channels,
            channels * 2,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
        y_emb: torch.Tensor,
        context: torch.Tensor = None,
    ):
        """
        Args:
            x: Latent representation inherited from previous residual block
                with the shape of [B x C x T].
            y_emb: Embeddings with the shape of [B x C], which will be FILM on the x.
            context: Context with the shape of [B x ``d_context`` x T], default to None.
        """

        h = x + y_emb[..., None]

        if self.context:
            h = self.gau(h, context)
        else:
            h = self.gau(h)

        h = self.out_proj(h)
        res, skip = h.chunk(2, 1)

        return (res + x) / math.sqrt(2.0), skip
