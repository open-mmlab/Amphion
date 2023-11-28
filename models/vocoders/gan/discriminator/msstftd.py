# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is adopted from META's Encodec under MIT License
# https://github.com/facebookresearch/encodec

"""MS-STFT discriminator, provided here for reference."""

import typing as tp

import torchaudio
import torch
from torch import nn
from einops import rearrange

from modules.vocoder_blocks import *


FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.List = [1, 2, 4],
        stride: tp.Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None,
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
            )
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min(
            (filters_scale ** (len(dilations) + 1)) * self.filters, max_filters
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        """Discriminator STFT Module is the sub module of MultiScaleSTFTDiscriminator.

        Args:
            x (torch.Tensor): input tensor of shape [B, 1, Time]

        Returns:
            z: z is the output of the last convolutional layer of shape
            fmap: fmap is the list of feature maps of every convolutional layer of shape
        """
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, "b c w t -> b c t w")
        for i, layer in enumerate(self.convs):
            z = layer(z)

            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(
        self,
        cfg,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: tp.List[int] = [1024, 2048, 512],
        hop_lengths: tp.List[int] = [256, 512, 256],
        win_lengths: tp.List[int] = [1024, 2048, 512],
        **kwargs,
    ):
        self.cfg = cfg
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    filters=self.cfg.model.msstftd.filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=n_ffts[i],
                    win_length=win_lengths[i],
                    hop_length=hop_lengths[i],
                    **kwargs,
                )
                for i in range(len(n_ffts))
            ]
        )
        self.num_discriminators = len(self.discriminators)

    def forward(self, y, y_hat) -> DiscriminatorOutput:
        """Multi-Scale STFT (MS-STFT) discriminator.

        Args:
            x (torch.Tensor): input waveform

        Returns:
            logits: list of every discriminator's output
            fmaps: list of every discriminator's feature maps,
                each feature maps is a list of Discriminator STFT's every layer
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
