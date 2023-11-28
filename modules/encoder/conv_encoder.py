# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from modules.generic.conv import Conv1d


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, z_channels, spk_channels, num_dilation_layer=10):
        super(ConvEncoder, self).__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.spk_channels = spk_channels

        self.pre_process = Conv1d(in_channels, 512, kernel_size=3)

        self.dilated_conv_layers = nn.ModuleList()
        for i in range(num_dilation_layer):
            dilation = 2**i
            self.dilated_conv_layers.append(
                DilatedConvBlock(512, 512, z_channels, spk_channels, dilation)
            )

    def forward(self, inputs, z, s):
        inputs = inputs.transpose(1, 2)
        outputs = self.pre_process(inputs)
        print(inputs.shape)
        for layer in self.dilated_conv_layers:
            outputs = layer(outputs, z, s)

        encoder_outputs = outputs.transpose(1, 2)
        return encoder_outputs


class DilatedConvBlock(nn.Module):
    """A stack of dilated convolutions interspersed
    with batch normalisation and ReLU activations"""

    def __init__(self, in_channels, out_channels, z_channels, s_channels, dilation):
        super(DilatedConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.s_channels = s_channels

        self.conv1d = Conv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation
        )
        self.batch_layer = BatchNorm1dLayer(out_channels, s_channels, z_channels)

    def forward(self, inputs, z, s):
        outputs = self.conv1d(inputs)
        outputs = self.batch_layer(outputs, z, s)
        return F.relu(outputs)


class BatchNorm1dLayer(nn.Module):
    """The latents z and speaker embedding s modulate the scale and
    shift parameters of the batch normalisation layers"""

    def __init__(self, num_features, s_channels=128, z_channels=128):
        super().__init__()

        self.num_features = num_features
        self.s_channels = s_channels
        self.z_channels = z_channels
        self.batch_nrom = nn.BatchNorm1d(num_features, affine=False)

        self.scale_layer = spectral_norm(nn.Linear(z_channels, num_features))
        self.scale_layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.scale_layer.bias.data.zero_()  # Initialise bias at 0

        self.shift_layer = spectral_norm(nn.Linear(s_channels, num_features))
        self.shift_layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.shift_layer.bias.data.zero_()  # Initialise bias at 0

    def forward(self, inputs, z, s):
        outputs = self.batch_nrom(inputs)
        scale = self.scale_layer(z)
        scale = scale.view(-1, self.num_features, 1)

        shift = self.shift_layer(s)
        shift = shift.view(-1, self.num_features, 1)

        outputs = scale * outputs + shift

        return outputs


if __name__ == "__main__":
    model = ConvEncoder(256, 64, 64)
    encoder_inputs = torch.randn(2, 256, 10)
    z = torch.randn(2, 64)
    speaker = torch.randn(1, 64)
    outputs, duration = model(encoder_inputs, z, speaker)
    print(outputs.shape, duration.shape)
