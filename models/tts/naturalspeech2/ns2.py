# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.tts.naturalspeech2.diffusion import Diffusion
from models.tts.naturalspeech2.diffusion_flow import DiffusionFlow
from models.tts.naturalspeech2.wavenet import WaveNet
from models.tts.naturalspeech2.prior_encoder import PriorEncoder
from modules.naturalpseech2.transformers import TransformerEncoder
from encodec import EncodecModel
from einops import rearrange, repeat

import os
import json


class NaturalSpeech2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.latent_dim = cfg.latent_dim
        self.query_emb_num = cfg.query_emb.query_token_num

        self.prior_encoder = PriorEncoder(cfg.prior_encoder)
        if cfg.diffusion.diffusion_type == "diffusion":
            self.diffusion = Diffusion(cfg.diffusion)
        elif cfg.diffusion.diffusion_type == "flow":
            self.diffusion = DiffusionFlow(cfg.diffusion)

        self.prompt_encoder = TransformerEncoder(cfg=cfg.prompt_encoder)
        if self.latent_dim != cfg.prompt_encoder.encoder_hidden:
            self.prompt_lin = nn.Linear(
                self.latent_dim, cfg.prompt_encoder.encoder_hidden
            )
            self.prompt_lin.weight.data.normal_(0.0, 0.02)
        else:
            self.prompt_lin = None

        self.query_emb = nn.Embedding(self.query_emb_num, cfg.query_emb.hidden_size)
        self.query_attn = nn.MultiheadAttention(
            cfg.query_emb.hidden_size, cfg.query_emb.head_num, batch_first=True
        )

        codec_model = EncodecModel.encodec_model_24khz()
        codec_model.set_target_bandwidth(12.0)
        codec_model.requires_grad_(False)
        self.quantizer = codec_model.quantizer

    @torch.no_grad()
    def code_to_latent(self, code):
        latent = self.quantizer.decode(code.transpose(0, 1))
        return latent

    def latent_to_code(self, latent, nq=16):
        residual = latent
        all_indices = []
        all_dist = []
        for i in range(nq):
            layer = self.quantizer.vq.layers[i]
            x = rearrange(residual, "b d n -> b n d")
            x = layer.project_in(x)
            shape = x.shape
            x = layer._codebook.preprocess(x)
            embed = layer._codebook.embed.t()
            dist = -(
                x.pow(2).sum(1, keepdim=True)
                - 2 * x @ embed
                + embed.pow(2).sum(0, keepdim=True)
            )
            indices = dist.max(dim=-1).indices
            indices = layer._codebook.postprocess_emb(indices, shape)
            dist = dist.reshape(*shape[:-1], dist.shape[-1])
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
            all_dist.append(dist)

        out_indices = torch.stack(all_indices)
        out_dist = torch.stack(all_dist)

        return out_indices, out_dist  # (nq, B, T); (nq, B, T, 1024)

    @torch.no_grad()
    def latent_to_latent(self, latent, nq=16):
        codes, _ = self.latent_to_code(latent, nq)
        latent = self.quantizer.vq.decode(codes)
        return latent

    def forward(
        self,
        code=None,
        pitch=None,
        duration=None,
        phone_id=None,
        phone_id_frame=None,
        frame_nums=None,
        ref_code=None,
        ref_frame_nums=None,
        phone_mask=None,
        mask=None,
        ref_mask=None,
    ):
        ref_latent = self.code_to_latent(ref_code)
        latent = self.code_to_latent(code)

        if self.latent_dim is not None:
            ref_latent = self.prompt_lin(ref_latent.transpose(1, 2))

        ref_latent = self.prompt_encoder(ref_latent, ref_mask, condition=None)
        spk_emb = ref_latent.transpose(1, 2)  # (B, d, T')

        spk_query_emb = self.query_emb(
            torch.arange(self.query_emb_num).to(latent.device)
        ).repeat(
            latent.shape[0], 1, 1
        )  # (B, query_emb_num, d)
        spk_query_emb, _ = self.query_attn(
            spk_query_emb,
            spk_emb.transpose(1, 2),
            spk_emb.transpose(1, 2),
            key_padding_mask=~(ref_mask.bool()),
        )  # (B, query_emb_num, d)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=mask,
            ref_emb=spk_emb,
            ref_mask=ref_mask,
            is_inference=False,
        )
        prior_condition = prior_out["prior_out"]  # (B, T, d)

        diff_out = self.diffusion(latent, mask, prior_condition, spk_query_emb)

        return diff_out, prior_out

    @torch.no_grad()
    def inference(
        self, ref_code=None, phone_id=None, ref_mask=None, inference_steps=1000
    ):
        ref_latent = self.code_to_latent(ref_code)

        if self.latent_dim is not None:
            ref_latent = self.prompt_lin(ref_latent.transpose(1, 2))

        ref_latent = self.prompt_encoder(ref_latent, ref_mask, condition=None)
        spk_emb = ref_latent.transpose(1, 2)  # (B, d, T')

        spk_query_emb = self.query_emb(
            torch.arange(self.query_emb_num).to(ref_latent.device)
        ).repeat(
            ref_latent.shape[0], 1, 1
        )  # (B, query_emb_num, d)
        spk_query_emb, _ = self.query_attn(
            spk_query_emb,
            spk_emb.transpose(1, 2),
            spk_emb.transpose(1, 2),
            key_padding_mask=~(ref_mask.bool()),
        )  # (B, query_emb_num, d)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=None,
            pitch=None,
            phone_mask=None,
            mask=None,
            ref_emb=spk_emb,
            ref_mask=ref_mask,
            is_inference=True,
        )
        prior_condition = prior_out["prior_out"]  # (B, T, d)

        z = torch.randn(
            prior_condition.shape[0], self.latent_dim, prior_condition.shape[1]
        ).to(ref_latent.device) / (1.20)
        x0 = self.diffusion.reverse_diffusion(
            z, None, prior_condition, inference_steps, spk_query_emb
        )

        return x0, prior_out

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self,
        code=None,
        pitch=None,
        duration=None,
        phone_id=None,
        ref_code=None,
        phone_mask=None,
        mask=None,
        ref_mask=None,
        n_timesteps=None,
        t=None,
    ):
        # o Only for debug

        ref_latent = self.code_to_latent(ref_code)
        latent = self.code_to_latent(code)

        if self.latent_dim is not None:
            ref_latent = self.prompt_lin(ref_latent.transpose(1, 2))

        ref_latent = self.prompt_encoder(ref_latent, ref_mask, condition=None)
        spk_emb = ref_latent.transpose(1, 2)  # (B, d, T')

        spk_query_emb = self.query_emb(
            torch.arange(self.query_emb_num).to(latent.device)
        ).repeat(
            latent.shape[0], 1, 1
        )  # (B, query_emb_num, d)
        spk_query_emb, _ = self.query_attn(
            spk_query_emb,
            spk_emb.transpose(1, 2),
            spk_emb.transpose(1, 2),
            key_padding_mask=~(ref_mask.bool()),
        )  # (B, query_emb_num, d)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=mask,
            ref_emb=spk_emb,
            ref_mask=ref_mask,
            is_inference=False,
        )
        prior_condition = prior_out["prior_out"]  # (B, T, d)

        diffusion_step = (
            torch.ones(
                latent.shape[0],
                dtype=latent.dtype,
                device=latent.device,
                requires_grad=False,
            )
            * t
        )
        diffusion_step = torch.clamp(diffusion_step, 1e-5, 1.0 - 1e-5)
        xt, _ = self.diffusion.forward_diffusion(
            x0=latent, diffusion_step=diffusion_step
        )
        # print(torch.abs(xt-latent).max(), torch.abs(xt-latent).mean(), torch.abs(xt-latent).std())

        x0 = self.diffusion.reverse_diffusion_from_t(
            xt, mask, prior_condition, n_timesteps, spk_query_emb, t_start=t
        )

        return x0, prior_out, xt
