# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""MelGAN Modules."""

import logging

import numpy as np
import torch

from models.tts.UniCATS.CTXvec2wav.build_model.layers import CausalConv1d
from models.tts.UniCATS.CTXvec2wav.build_model.layers import CausalConvTranspose1d
from models.tts.UniCATS.CTXvec2wav.build_model.layers import ResidualStack
from models.tts.UniCATS.CTXvec2wav.utils import read_hdf5


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        kernel_size=7,
        channels=512,
        bias=True,
        upsample_scales=[8, 8, 2, 2],
        stack_kernel_size=3,
        stacks=3,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_final_nonlinear_activation=True,
        use_weight_norm=True,
        use_causal_conv=False,
    ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANGenerator, self).__init__()

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        # add initial layer
        layers = []
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(
                    in_channels,
                    channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            if not use_causal_conv:
                layers += [
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias=bias,
                    )
                ]
            else:
                layers += [
                    CausalConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias,
                    )
                ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size ** j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

        # add final layer
        layers += [
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        ]
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias
                ),
            ]
        else:
            layers += [
                CausalConv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
            ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        # initialize pqmf for inference
        self.pqmf = None

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        return self.melgan(c)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")

    def inference(self, c, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        # if not isinstance(c, torch.Tensor):
        #     c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.melgan(c.transpose(1, 2))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)


class MelGANDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=[5, 3],
        channels=16,
        max_downsample_channels=1024,
        bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
    ):
        """Initilize MelGAN discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.

        """
        super(MelGANDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    in_channels, channels, np.prod(kernel_sizes), bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of output tensors of each layer.

        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs


class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        scales=3,
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        },
        kernel_sizes=[5, 3],
        channels=16,
        max_downsample_channels=1024,
        bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_weight_norm=True,
    ):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANMultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for _ in range(scales):
            self.discriminators += [
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                )
            ]
        self.pooling = getattr(torch.nn, downsample_pooling)(
            **downsample_pooling_params
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
