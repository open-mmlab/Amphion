# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math

from torch import nn
from torch.nn import functional as F

from .conv import Conv1d as conv_Conv1d


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    m = conv_Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def Conv1d1x1(in_channels, out_channels, bias=True):
    return Conv1d(
        in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
    )


def _conv1x1_forward(conv, x, is_incremental):
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit

    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(
        self,
        residual_channels,
        gate_channels,
        kernel_size,
        skip_out_channels=None,
        cin_channels=-1,
        dropout=1 - 0.95,
        padding=None,
        dilation=1,
        causal=True,
        bias=True,
        *args,
        **kwargs,
    ):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout

        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
            *args,
            **kwargs,
        )

        # mel conditioning
        self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=False)

        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias)

    def forward(self, x, c=None):
        return self._forward(x, c, False)

    def incremental_forward(self, x, c=None):
        return self._forward(x, c, True)

    def clear_buffer(self):
        for c in [
            self.conv,
            self.conv1x1_out,
            self.conv1x1_skip,
            self.conv1x1c,
        ]:
            if c is not None:
                c.clear_buffer()

    def _forward(self, x, c, is_incremental):
        """Forward

        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Mel conditioning features
        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, : residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        assert self.conv1x1c is not None
        c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
        ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
        a, b = a + ca, b + cb

        x = torch.tanh(a) * torch.sigmoid(b)

        # For skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

        # For residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

        x = (x + residual) * math.sqrt(0.5)
        return x, s
