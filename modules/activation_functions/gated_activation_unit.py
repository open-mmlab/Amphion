# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from modules.general.utils import Conv1d


class GaU(nn.Module):
    r"""Gated Activation Unit (GaU) proposed in `Gated Activation Units for Neural
    Networks <https://arxiv.org/pdf/1606.05328.pdf>`_.

    Args:
        channels: number of input channels.
        kernel_size: kernel size of the convolution.
        dilation: dilation rate of the convolution.
        d_context: dimension of context tensor, None if don't use context.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        d_context: int = None,
    ):
        super().__init__()

        self.context = d_context

        self.conv = Conv1d(
            channels,
            channels * 2,
            kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
        )

        if self.context:
            self.context_proj = Conv1d(d_context, channels * 2, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        r"""Calculate forward propagation.

        Args:
            x: input tensor with shape [B, C, T].
            context: context tensor with shape [B, ``d_context``, T], default to None.
        """

        h = self.conv(x)

        if self.context:
            h = h + self.context_proj(context)

        h1, h2 = h.chunk(2, 1)
        h = torch.tanh(h1) * torch.sigmoid(h2)

        return h
