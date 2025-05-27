# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .cnn import ConvNeXtBlock
from .dac_model import DAC
import torch.nn as nn
import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

# from .base import CodecMixin
from .dac_layers import Snake1d
from .dac_layers import WNConv1d
from .dac_layers import WNConvTranspose1d
from .dac_quantize import ResidualVectorQuantize
from easydict import EasyDict as edict
import torch.nn.functional as F
import random
from einops import rearrange


class DualCodec(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 16384,
        codebook_dim: Union[int, list] = 8,
        semantic_codebook_dim=8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        distill_projection_out_dim=1024,
        convnext_dim=768,
        convnext_layers=4,
        decode_semantic_for_codec=True,
        is_causal=False,
        semantic_downsample_factor=2,
    ):
        self.semantic_downsample_factor = semantic_downsample_factor
        super().__init__()

        self.dac = DAC(
            encoder_dim,
            encoder_rates,
            latent_dim,
            decoder_dim,
            decoder_rates,
            n_codebooks,
            codebook_size,
            codebook_dim,
            quantizer_dropout,
            sample_rate,
            distill_projection_out_dim,
            distill=False,
        )
        self.decode_semantic_for_codec = decode_semantic_for_codec
        self.encoder_rates = encoder_rates
        self.convnext_encoder = nn.Sequential(
            WNConv1d(
                1024,
                convnext_dim,
                kernel_size=1,
            ),
            *[
                ConvNeXtBlock(
                    dim=convnext_dim, intermediate_dim=2048, is_causal=is_causal
                )
                for _ in range(convnext_layers)
            ],  # Unpack the list directly into nn.Sequential
        )
        self.semantic_vq = ResidualVectorQuantize(
            convnext_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=semantic_codebook_dim,
        )
        self.convnext_decoder = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=convnext_dim,
                    intermediate_dim=2048,
                    is_causal=is_causal,
                )
                for _ in range(convnext_layers)
            ],  # Unpack the list directly into nn.Sequential
            WNConv1d(
                convnext_dim,
                1024,
                kernel_size=1,
            ),
        )
        if not self.decode_semantic_for_codec:
            assert convnext_dim == 1024

    def semantic_quantize(self, semantic_repr):
        semantic = self.convnext_encoder(semantic_repr)
        (
            semantic,
            codes,
            latents,
            commitment_loss,
            codebook_loss,
            first_layer_quantized,
        ) = self.semantic_vq(semantic)
        codes = rearrange(codes, "b 1 t -> b t")
        return codes

    def encode(
        self, audio_data, num_quantizers=None, sample_rate=24000, semantic_repr=None
    ):
        assert not self.training
        semantic = self.convnext_encoder(semantic_repr)
        (
            semantic,
            codes,
            latents,
            commitment_loss,
            codebook_loss,
            first_layer_quantized,
        ) = self.semantic_vq(semantic)
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)
        semantic_codes = codes

        if num_quantizers == 1:
            return semantic_codes, None

        if num_quantizers is not None:
            num_quantizers -= 1

        acoustic_codes = self.dac.encode(
            audio_data,
            sample_rate=sample_rate,
            n_quantizers=num_quantizers,
            subtracted_latent=semantic,
        )[1]
        return semantic_codes, acoustic_codes  # [B, n_q, T]

    @torch.no_grad()
    def decode_from_codes(self, semantic_codes, acoustic_codes):
        """both [B, n_q, T]"""
        semantic = self.semantic_vq.from_codes(semantic_codes)[0]
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        audio = self.dac.decode_from_codes(acoustic_codes, semantic)
        return audio

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = 24000,
        n_quantizers: int = None,
        semantic_repr=None,
        bypass_quantize_rate=0.125,
        possibly_no_quantizer=False,
    ):
        semantic = self.convnext_encoder(semantic_repr)
        (
            semantic,
            codes,
            latents,
            commitment_loss,
            codebook_loss,
            first_layer_quantized,
        ) = self.semantic_vq(semantic)
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        bypass_quantize = random.random() < bypass_quantize_rate
        if not self.training:
            bypass_quantize = False
        if n_quantizers == 1:
            bypass_quantize = True
        if n_quantizers is not None:
            n_quantizers = n_quantizers - 1
        acoustic_edict = self.dac(
            audio_data,
            sample_rate,
            n_quantizers,
            subtracted_latent=semantic,
            bypass_quantize=bypass_quantize,
            possibly_no_quantizer=possibly_no_quantizer,
        )
        if not self.decode_semantic_for_codec:
            # decode afterwards
            semantic = self.convnext_decoder(semantic)

        semantic_edict = edict(
            {
                "x": semantic,
                "codes": codes,
                "latents": latents,
                "penalty": commitment_loss,
                "vq/codebook_loss": codebook_loss,
                "metrics": {},
                "bypassed_quantize": bypass_quantize,
                # "first_layer_quantized": first_layer_quantized,
            }
        )
        return acoustic_edict, semantic_edict
