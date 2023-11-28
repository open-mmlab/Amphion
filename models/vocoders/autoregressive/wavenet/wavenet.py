# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from torch import nn
from torch.nn import functional as F

from .modules import Conv1d1x1, ResidualConv1dGLU
from .upsample import ConvInUpsampleNetwork


def receptive_field_size(
    total_layers, num_cycles, kernel_size, dilation=lambda x: 2**x
):
    """Compute receptive field size

    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.

    Returns:
        int: receptive field size in sample

    """
    assert total_layers % num_cycles == 0

    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


class WaveNet(nn.Module):
    """The WaveNet model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        input_dim (int): Number of mel-spec dimension.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected..
    """

    def __init__(self, cfg):
        super(WaveNet, self).__init__()
        self.cfg = cfg
        self.scalar_input = self.cfg.VOCODER.SCALAR_INPUT
        self.out_channels = self.cfg.VOCODER.OUT_CHANNELS
        self.cin_channels = self.cfg.VOCODER.INPUT_DIM
        self.residual_channels = self.cfg.VOCODER.RESIDUAL_CHANNELS
        self.layers = self.cfg.VOCODER.LAYERS
        self.stacks = self.cfg.VOCODER.STACKS
        self.gate_channels = self.cfg.VOCODER.GATE_CHANNELS
        self.kernel_size = self.cfg.VOCODER.KERNEL_SIZE
        self.skip_out_channels = self.cfg.VOCODER.SKIP_OUT_CHANNELS
        self.dropout = self.cfg.VOCODER.DROPOUT
        self.upsample_scales = self.cfg.VOCODER.UPSAMPLE_SCALES
        self.mel_frame_pad = self.cfg.VOCODER.MEL_FRAME_PAD

        assert self.layers % self.stacks == 0

        layers_per_stack = self.layers // self.stacks
        if self.scalar_input:
            self.first_conv = Conv1d1x1(1, self.residual_channels)
        else:
            self.first_conv = Conv1d1x1(self.out_channels, self.residual_channels)

        self.conv_layers = nn.ModuleList()
        for layer in range(self.layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualConv1dGLU(
                self.residual_channels,
                self.gate_channels,
                kernel_size=self.kernel_size,
                skip_out_channels=self.skip_out_channels,
                bias=True,
                dilation=dilation,
                dropout=self.dropout,
                cin_channels=self.cin_channels,
            )
            self.conv_layers.append(conv)

        self.last_conv_layers = nn.ModuleList(
            [
                nn.ReLU(inplace=True),
                Conv1d1x1(self.skip_out_channels, self.skip_out_channels),
                nn.ReLU(inplace=True),
                Conv1d1x1(self.skip_out_channels, self.out_channels),
            ]
        )

        self.upsample_net = ConvInUpsampleNetwork(
            upsample_scales=self.upsample_scales,
            cin_pad=self.mel_frame_pad,
            cin_channels=self.cin_channels,
        )

        self.receptive_field = receptive_field_size(
            self.layers, self.stacks, self.kernel_size
        )

    def forward(self, x, mel, softmax=False):
        """Forward step

        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            mel (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            softmax (bool): Whether applies softmax or not.

        Returns:
            Tensor: output, shape B x out_channels x T
        """
        B, _, T = x.size()

        mel = self.upsample_net(mel)
        assert mel.shape[-1] == x.shape[-1]

        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, mel)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        x = F.softmax(x, dim=1) if softmax else x

        return x

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
