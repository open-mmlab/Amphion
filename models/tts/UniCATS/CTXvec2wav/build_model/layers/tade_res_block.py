# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""StyleMelGAN's TADEResBlock Modules."""

from functools import partial

import torch


class TADELayer(torch.nn.Module):
    """TADE Layer module."""

    def __init__(
        self,
        in_channels=64,
        aux_channels=80,
        kernel_size=9,
        bias=True,
        upsample_factor=2,
        upsample_mode="nearest",
    ):
        """Initilize TADE layer."""
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(in_channels)
        self.aux_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                aux_channels,
                in_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.gated_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                in_channels * 2,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').

        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled aux tensor (B, in_channels, T * aux_upsample_factor).

        """
        x = self.norm(x)
        c = self.upsample(c)
        c = self.aux_conv(c)
        cg = self.gated_conv(c)
        cg1, cg2 = cg.split(cg.size(1) // 2, dim=1)
        # NOTE(kan-bayashi): Use upsample for noise input here?
        y = cg1 * self.upsample(x) + cg2
        # NOTE(kan-bayashi): Return upsampled aux here?
        return y, c


class TADEResBlock(torch.nn.Module):
    """TADEResBlock module."""

    def __init__(
        self,
        in_channels=64,
        aux_channels=80,
        kernel_size=9,
        dilation=2,
        bias=True,
        upsample_factor=2,
        upsample_mode="nearest",
        gated_function="softmax",
    ):
        """Initialize TADEResBlock module."""
        super().__init__()
        self.tade1 = TADELayer(
            in_channels=in_channels,
            aux_channels=aux_channels,
            kernel_size=kernel_size,
            bias=bias,
            # NOTE(kan-bayashi): Use upsample in the first TADE layer?
            upsample_factor=1,
            upsample_mode=upsample_mode,
        )
        self.gated_conv1 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )
        self.tade2 = TADELayer(
            in_channels=in_channels,
            aux_channels=in_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsample_factor=upsample_factor,
            upsample_mode=upsample_mode,
        )
        self.gated_conv2 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation,
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )
        if gated_function == "softmax":
            self.gated_function = partial(torch.softmax, dim=1)
        elif gated_function == "sigmoid":
            self.gated_function = torch.sigmoid
        else:
            raise ValueError(f"{gated_function} is not supported.")

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').

        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled auxirialy tensor (B, in_channels, T * in_upsample_factor).

        """
        residual = x

        x, c = self.tade1(x, c)
        x = self.gated_conv1(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = self.gated_function(xa) * torch.tanh(xb)

        x, c = self.tade2(x, c)
        x = self.gated_conv2(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = self.gated_function(xa) * torch.tanh(xb)

        # NOTE(kan-bayashi): Return upsampled aux here?
        return self.upsample(residual) + x, c
