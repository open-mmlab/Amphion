# -*- coding: utf-8 -*-

"""Upsampling module.

This code is modified from https://github.com/r9y9/wavenet_vocoder.

"""

import numpy as np
import torch
import torch.nn.functional as F

from models.tts.UniCATS.CTXvec2wav.build_model.layers import Conv1d


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode="nearest"):
        """Initialize Stretch2d module.

        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.

        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, C, F, T).

        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),

        """
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode
        )


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        self.weight.data.fill_(1.0 / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(
        self,
        upsample_scales,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode="nearest",
        freq_axis_kernel_size=1,
        use_causal_conv=False,
    ):
        """Initialize upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            interpolate_mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.

        """
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            # interpolation layer
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]

            # conv layer
            assert (
                freq_axis_kernel_size - 1
            ) % 2 == 0, "Not support even number freq axis kernel size."
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]

            # nonlinear
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params
                )
                self.up_layers += [nonlinear]

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T).

        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).

        """
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., : c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)  # (B, C, T')


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(
        self,
        upsample_scales,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode="nearest",
        freq_axis_kernel_size=1,
        aux_channels=80,
        aux_context_window=0,
        use_causal_conv=False,
    ):
        """Initialize convolution + upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_channels (int): Number of channels of pre-convolutional layer.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_causal_conv (bool): Whether to use causal structure.

        """
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        # To capture wide-context information in conditional features
        kernel_size = (
            aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        )
        # NOTE(kan-bayashi): Here do not use padding because the input is already padded
        self.conv_in = Conv1d(
            aux_channels, aux_channels, kernel_size=kernel_size, bias=False
        )
        self.upsample = UpsampleNetwork(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
        )

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T').

        Returns:
            Tensor: Upsampled tensor (B, C, T),
                where T = (T' - aux_context_window * 2) * prod(upsample_scales).

        Note:
            The length of inputs considers the context window size.

        """
        c_ = self.conv_in(c)
        c = c_[:, :, : -self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)
