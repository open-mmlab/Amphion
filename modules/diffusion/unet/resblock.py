# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import UNetBlock
from modules.general.utils import (
    append_dims,
    ConvNd,
    normalization,
    zero_module,
)


class ResBlock(UNetBlock):
    r"""A residual block that can optionally change the number of channels.

    Args:
        channels: the number of input channels.
        emb_channels: the number of timestep embedding channels.
        dropout: the rate of dropout.
        out_channels: if specified, the number of out channels.
        use_conv: if True and out_channels is specified, use a spatial
            convolution instead of a smaller 1x1 convolution to change the
            channels in the skip connection.
        dims: determines if the signal is 1D, 2D, or 3D.
        up: if True, use this block for upsampling.
        down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout: float = 0.0,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            ConvNd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            ConvNd(
                dims,
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                1,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                ConvNd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = ConvNd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = ConvNd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

            x: an [N x C x ...] Tensor of features.
            emb: an [N x emb_channels x ...] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        emb_out = append_dims(emb_out, h.dim())
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Upsample(nn.Module):
    r"""An upsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
            upsampling occurs in the inner-two dimensions.
        out_channels: if specified, the number of out channels.
    """

    def __init__(self, channels, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims
        self.conv = ConvNd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    r"""A downsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
            downsampling occurs in the inner-two dimensions.
        out_channels: if specified, the number of output channels.
    """

    def __init__(self, channels, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = ConvNd(
            dims, self.channels, self.out_channels, 3, stride=stride, padding=1
        )

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
