# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from modules.vocoder_blocks import *

LRELU_SLOPE = 0.1


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, spec: torch.Tensor, window) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


# The ASP and PSP Module are adopted from APNet under the MIT License
# https://github.com/YangAi520/APNet/blob/main/models.py


class ASPResBlock(torch.nn.Module):
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ASPResBlock, self).__init__()
        self.cfg = cfg
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class PSPResBlock(torch.nn.Module):
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(PSPResBlock, self).__init__()
        self.cfg = cfg
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class APNet(torch.nn.Module):
    def __init__(self, cfg):
        super(APNet, self).__init__()
        self.cfg = cfg
        self.ASP_num_kernels = len(cfg.model.apnet.ASP_resblock_kernel_sizes)
        self.PSP_num_kernels = len(cfg.model.apnet.PSP_resblock_kernel_sizes)

        self.ASP_input_conv = weight_norm(
            Conv1d(
                cfg.preprocess.n_mel,
                cfg.model.apnet.ASP_channel,
                cfg.model.apnet.ASP_input_conv_kernel_size,
                1,
                padding=get_padding(cfg.model.apnet.ASP_input_conv_kernel_size, 1),
            )
        )
        self.PSP_input_conv = weight_norm(
            Conv1d(
                cfg.preprocess.n_mel,
                cfg.model.apnet.PSP_channel,
                cfg.model.apnet.PSP_input_conv_kernel_size,
                1,
                padding=get_padding(cfg.model.apnet.PSP_input_conv_kernel_size, 1),
            )
        )

        self.ASP_ResNet = nn.ModuleList()
        for j, (k, d) in enumerate(
            zip(
                cfg.model.apnet.ASP_resblock_kernel_sizes,
                cfg.model.apnet.ASP_resblock_dilation_sizes,
            )
        ):
            self.ASP_ResNet.append(ASPResBlock(cfg, cfg.model.apnet.ASP_channel, k, d))

        self.PSP_ResNet = nn.ModuleList()
        for j, (k, d) in enumerate(
            zip(
                cfg.model.apnet.PSP_resblock_kernel_sizes,
                cfg.model.apnet.PSP_resblock_dilation_sizes,
            )
        ):
            self.PSP_ResNet.append(PSPResBlock(cfg, cfg.model.apnet.PSP_channel, k, d))

        self.ASP_output_conv = weight_norm(
            Conv1d(
                cfg.model.apnet.ASP_channel,
                cfg.preprocess.n_fft // 2 + 1,
                cfg.model.apnet.ASP_output_conv_kernel_size,
                1,
                padding=get_padding(cfg.model.apnet.ASP_output_conv_kernel_size, 1),
            )
        )
        self.PSP_output_R_conv = weight_norm(
            Conv1d(
                cfg.model.apnet.PSP_channel,
                cfg.preprocess.n_fft // 2 + 1,
                cfg.model.apnet.PSP_output_R_conv_kernel_size,
                1,
                padding=get_padding(cfg.model.apnet.PSP_output_R_conv_kernel_size, 1),
            )
        )
        self.PSP_output_I_conv = weight_norm(
            Conv1d(
                cfg.model.apnet.PSP_channel,
                cfg.preprocess.n_fft // 2 + 1,
                cfg.model.apnet.PSP_output_I_conv_kernel_size,
                1,
                padding=get_padding(cfg.model.apnet.PSP_output_I_conv_kernel_size, 1),
            )
        )

        self.iSTFT = ISTFT(
            self.cfg.preprocess.n_fft,
            hop_length=self.cfg.preprocess.hop_size,
            win_length=self.cfg.preprocess.win_size,
        )

        self.ASP_output_conv.apply(init_weights)
        self.PSP_output_R_conv.apply(init_weights)
        self.PSP_output_I_conv.apply(init_weights)

    def forward(self, mel):
        logamp = self.ASP_input_conv(mel)
        logamps = None
        for j in range(self.ASP_num_kernels):
            if logamps is None:
                logamps = self.ASP_ResNet[j](logamp)
            else:
                logamps += self.ASP_ResNet[j](logamp)
        logamp = logamps / self.ASP_num_kernels
        logamp = F.leaky_relu(logamp)
        logamp = self.ASP_output_conv(logamp)

        pha = self.PSP_input_conv(mel)
        phas = None
        for j in range(self.PSP_num_kernels):
            if phas is None:
                phas = self.PSP_ResNet[j](pha)
            else:
                phas += self.PSP_ResNet[j](pha)
        pha = phas / self.PSP_num_kernels
        pha = F.leaky_relu(pha)
        R = self.PSP_output_R_conv(pha)
        I = self.PSP_output_I_conv(pha)

        pha = torch.atan2(I, R)

        rea = torch.exp(logamp) * torch.cos(pha)
        imag = torch.exp(logamp) * torch.sin(pha)

        spec = torch.cat((rea.unsqueeze(-1), imag.unsqueeze(-1)), -1)

        spec = torch.view_as_complex(spec)

        audio = self.iSTFT.forward(
            spec, torch.hann_window(self.cfg.preprocess.win_size).to(mel.device)
        )

        return logamp, pha, rea, imag, audio.unsqueeze(1)
