# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow Layer modules complatible with pytorch."""

import tensorflow as tf


class TFReflectionPad1d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size):
        """Initialize TFReflectionPad1d module.

        Args:
            padding_size (int): Padding size.

        """
        super(TFReflectionPad1d, self).__init__()
        self.padding_size = padding_size

    @tf.function
    def call(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, T, 1, C).

        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, 1, C).

        """
        return tf.pad(
            x,
            [[0, 0], [self.padding_size, self.padding_size], [0, 0], [0, 0]],
            "REFLECT",
        )


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(self, channels, kernel_size, stride, padding):
        """Initialize TFConvTranspose1d( module.

        Args:
            channels (int): Number of channels.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").

        """
        super(TFConvTranspose1d, self).__init__()
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=channels,
            kernel_size=(kernel_size, 1),
            strides=(stride, 1),
            padding=padding,
        )

    @tf.function
    def call(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, T, 1, C).

        Returns:
            Tensors: Output tensor (B, T', 1, C').

        """
        x = self.conv1d_transpose(x)
        return x


class TFResidualStack(tf.keras.layers.Layer):
    """Tensorflow ResidualStack module."""

    def __init__(
        self,
        kernel_size,
        channels,
        dilation,
        bias,
        nonlinear_activation,
        nonlinear_activation_params,
        padding,
    ):
        """Initialize TFResidualStack module.

        Args:
            kernel_size (int): Kernel size.
            channles (int): Number of channels.
            dilation (int): Dilation ine.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            padding (str): Padding type ("same" or "valid").

        """
        super(TFResidualStack, self).__init__()
        self.block = [
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            TFReflectionPad1d(dilation),
            tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=(kernel_size, 1),
                dilation_rate=(dilation, 1),
                use_bias=bias,
                padding="valid",
            ),
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            tf.keras.layers.Conv2D(filters=channels, kernel_size=1, use_bias=bias),
        ]
        self.shortcut = tf.keras.layers.Conv2D(
            filters=channels, kernel_size=1, use_bias=bias
        )

    @tf.function
    def call(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, T, 1, C).

        Returns:
            Tensor: Output tensor (B, T, 1, C).

        """
        _x = tf.identity(x)
        for i, layer in enumerate(self.block):
            _x = layer(_x)
        shortcut = self.shortcut(x)
        return shortcut + _x
