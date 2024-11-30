# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import json5
from librosa.filters import mel as librosa_mel_fn
from einops.layers.torch import Rearrange


class Diffusion(nn.Module):
    def __init__(self, cfg, diff_model):
        super().__init__()

        self.cfg = cfg
        self.diff_estimator = diff_model
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max
        self.sigma = cfg.sigma
        self.noise_factor = cfg.noise_factor

    def forward(self, x, condition_embedding, x_mask, reference_embedding, offset=1e-5):
        diffusion_step = torch.rand(
            x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False
        )
        diffusion_step = torch.clamp(diffusion_step, offset, 1.0 - offset)
        xt, z = self.forward_diffusion(x0=x, diffusion_step=diffusion_step)

        cum_beta = self.get_cum_beta(diffusion_step.unsqueeze(-1).unsqueeze(-1))
        x0_pred = self.diff_estimator(
            xt, condition_embedding, x_mask, reference_embedding, diffusion_step
        )
        mean_pred = x0_pred * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        variance = (self.sigma**2) * (1.0 - torch.exp(-cum_beta / (self.sigma**2)))
        noise_pred = (xt - mean_pred) / (torch.sqrt(variance) * self.noise_factor)
        noise = z
        diff_out = {"x0_pred": x0_pred, "noise_pred": noise_pred, "noise": noise}
        return diff_out

    @torch.no_grad()
    def get_cum_beta(self, time_step):
        return self.beta_min * time_step + 0.5 * (self.beta_max - self.beta_min) * (
            time_step**2
        )

    @torch.no_grad()
    def get_beta_t(self, time_step):
        return self.beta_min + (self.beta_max - self.beta_min) * time_step

    @torch.no_grad()
    def forward_diffusion(self, x0, diffusion_step):
        time_step = diffusion_step.unsqueeze(-1).unsqueeze(-1)
        cum_beta = self.get_cum_beta(time_step)
        mean = x0 * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        variance = (self.sigma**2) * (1 - torch.exp(-cum_beta / (self.sigma**2)))
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean + z * torch.sqrt(variance) * self.noise_factor
        return xt, z

    @torch.no_grad()
    def cal_dxt(
        self, xt, condition_embedding, x_mask, reference_embedding, diffusion_step, h
    ):
        time_step = diffusion_step.unsqueeze(-1).unsqueeze(-1)
        cum_beta = self.get_cum_beta(time_step=time_step)
        beta_t = self.get_beta_t(time_step=time_step)
        x0_pred = self.diff_estimator(
            xt, condition_embedding, x_mask, reference_embedding, diffusion_step
        )
        mean_pred = x0_pred * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        noise_pred = xt - mean_pred
        variance = (self.sigma**2) * (1.0 - torch.exp(-cum_beta / (self.sigma**2)))
        logp = -noise_pred / (variance + 1e-8)
        dxt = -0.5 * h * beta_t * (logp + xt / (self.sigma**2))
        return dxt

    @torch.no_grad()
    def reverse_diffusion(
        self, z, condition_embedding, x_mask, reference_embedding, n_timesteps
    ):
        h = 1.0 / max(n_timesteps, 1)
        xt = z
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            dxt = self.cal_dxt(
                xt,
                condition_embedding,
                x_mask,
                reference_embedding,
                diffusion_step=t,
                h=h,
            )
            xt_ = xt - dxt
            if self.cfg.ode_solve_method == "midpoint":
                x_mid = 0.5 * (xt_ + xt)
                dxt = self.cal_dxt(
                    x_mid,
                    condition_embedding,
                    x_mask,
                    reference_embedding,
                    diffusion_step=t + 0.5 * h,
                    h=h,
                )
                xt = xt - dxt
            elif self.cfg.ode_solve_method == "euler":
                xt = xt_
        return xt

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self, z, condition_embedding, x_mask, reference_embedding, n_timesteps, t_start
    ):
        h = t_start / max(n_timesteps, 1)
        xt = z
        for i in range(n_timesteps):
            t = (t_start - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            dxt = self.cal_dxt(
                xt,
                x_mask,
                condition_embedding,
                x_mask,
                reference_embedding,
                diffusion_step=t,
                h=h,
            )
            xt_ = xt - dxt
            if self.cfg.ode_solve_method == "midpoint":
                x_mid = 0.5 * (xt_ + xt)
                dxt = self.cal_dxt(
                    x_mid,
                    condition_embedding,
                    x_mask,
                    reference_embedding,
                    diffusion_step=t + 0.5 * h,
                    h=h,
                )
                xt = xt - dxt
            elif self.cfg.ode_solve_method == "euler":
                xt = xt_
        return xt


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Linear2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim * 2)
        self.linear_2 = nn.Linear(dim * 2, dim)
        self.linear_1.weight.data.normal_(0.0, 0.02)
        self.linear_2.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.in_dim = normalized_shape
        self.norm = nn.LayerNorm(self.in_dim, eps=eps, elementwise_affine=False)
        self.style = nn.Linear(self.in_dim, self.in_dim * 2)
        self.style.bias.data[: self.in_dim] = 1
        self.style.bias.data[self.in_dim :] = 0

    def forward(self, x, condition):
        # x: (B, T, d); condition: (B, T, d)

        style = self.style(torch.mean(condition, dim=1, keepdim=True))

        gamma, beta = style.chunk(2, -1)

        out = self.norm(x)

        out = gamma * out + beta
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

        self.dropout = dropout
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return F.dropout(x, self.dropout, training=self.training)


class TransformerFFNLayer(nn.Module):
    def __init__(
        self, encoder_hidden, conv_filter_size, conv_kernel_size, encoder_dropout
    ):
        super().__init__()

        self.encoder_hidden = encoder_hidden
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout

        self.ffn_1 = nn.Conv1d(
            self.encoder_hidden,
            self.conv_filter_size,
            self.conv_kernel_size,
            padding=self.conv_kernel_size // 2,
        )
        self.ffn_1.weight.data.normal_(0.0, 0.02)
        self.ffn_2 = nn.Linear(self.conv_filter_size, self.encoder_hidden)
        self.ffn_2.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        # x: (B, T, d)
        x = self.ffn_1(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (B, T, d) -> (B, d, T) -> (B, T, d)
        x = F.silu(x)
        x = F.dropout(x, self.encoder_dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class TransformerFFNLayerOld(nn.Module):
    def __init__(
        self, encoder_hidden, conv_filter_size, conv_kernel_size, encoder_dropout
    ):
        super().__init__()

        self.encoder_hidden = encoder_hidden
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout

        self.ffn_1 = nn.Linear(self.encoder_hidden, self.conv_filter_size)
        self.ffn_1.weight.data.normal_(0.0, 0.02)
        self.ffn_2 = nn.Linear(self.conv_filter_size, self.encoder_hidden)
        self.ffn_2.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        x = self.ffn_1(x)
        x = F.silu(x)
        x = F.dropout(x, self.encoder_dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        encoder_hidden,
        encoder_head,
        conv_filter_size,
        conv_kernel_size,
        encoder_dropout,
        use_cln,
        use_skip_connection,
        use_new_ffn,
        add_diff_step,
    ):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.encoder_head = encoder_head
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout
        self.use_cln = use_cln
        self.use_skip_connection = use_skip_connection
        self.use_new_ffn = use_new_ffn
        self.add_diff_step = add_diff_step

        if not self.use_cln:
            self.ln_1 = nn.LayerNorm(self.encoder_hidden)
            self.ln_2 = nn.LayerNorm(self.encoder_hidden)
        else:
            self.ln_1 = StyleAdaptiveLayerNorm(self.encoder_hidden)
            self.ln_2 = StyleAdaptiveLayerNorm(self.encoder_hidden)

        self.self_attn = nn.MultiheadAttention(
            self.encoder_hidden, self.encoder_head, batch_first=True
        )

        if self.use_new_ffn:
            self.ffn = TransformerFFNLayer(
                self.encoder_hidden,
                self.conv_filter_size,
                self.conv_kernel_size,
                self.encoder_dropout,
            )
        else:
            self.ffn = TransformerFFNLayerOld(
                self.encoder_hidden,
                self.conv_filter_size,
                self.conv_kernel_size,
                self.encoder_dropout,
            )

        if self.use_skip_connection:
            self.skip_linear = nn.Linear(self.encoder_hidden * 2, self.encoder_hidden)
            self.skip_linear.weight.data.normal_(0.0, 0.02)
            self.skip_layernorm = nn.LayerNorm(self.encoder_hidden)

        if self.add_diff_step:
            self.diff_step_emb = SinusoidalPosEmb(dim=self.encoder_hidden)
            # self.diff_step_projection = nn.linear(self.encoder_hidden, self.encoder_hidden)
            # self.encoder_hidden.weight.data.normal_(0.0, 0.02)
            self.diff_step_projection = Linear2(self.encoder_hidden)

    def forward(
        self, x, key_padding_mask, conditon=None, skip_res=None, diffusion_step=None
    ):
        # x: (B, T, d); key_padding_mask: (B, T), mask is 0; condition: (B, T, d); skip_res: (B, T, d); diffusion_step: (B,)

        if self.use_skip_connection and skip_res != None:
            x = torch.cat([x, skip_res], dim=-1)  # (B, T, 2*d)
            x = self.skip_linear(x)
            x = self.skip_layernorm(x)

        if self.add_diff_step and diffusion_step != None:
            diff_step_embedding = self.diff_step_emb(diffusion_step)
            diff_step_embedding = self.diff_step_projection(diff_step_embedding)
            x = x + diff_step_embedding.unsqueeze(1)

        residual = x

        # pre norm
        if self.use_cln:
            x = self.ln_1(x, conditon)
        else:
            x = self.ln_1(x)

        # self attention
        if key_padding_mask != None:
            key_padding_mask_input = ~(key_padding_mask.bool())
        else:
            key_padding_mask_input = None
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask_input
        )
        x = F.dropout(x, self.encoder_dropout, training=self.training)

        x = residual + x

        # pre norm
        residual = x
        if self.use_cln:
            x = self.ln_2(x, conditon)
        else:
            x = self.ln_2(x)

        # ffn
        x = self.ffn(x)

        x = residual + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        enc_emb_tokens=None,
        encoder_layer=None,
        encoder_hidden=None,
        encoder_head=None,
        conv_filter_size=None,
        conv_kernel_size=None,
        encoder_dropout=None,
        use_cln=None,
        use_skip_connection=None,
        use_new_ffn=None,
        add_diff_step=None,
        cfg=None,
    ):
        super().__init__()

        self.encoder_layer = (
            encoder_layer if encoder_layer is not None else cfg.encoder_layer
        )
        self.encoder_hidden = (
            encoder_hidden if encoder_hidden is not None else cfg.encoder_hidden
        )
        self.encoder_head = (
            encoder_head if encoder_head is not None else cfg.encoder_head
        )
        self.conv_filter_size = (
            conv_filter_size if conv_filter_size is not None else cfg.conv_filter_size
        )
        self.conv_kernel_size = (
            conv_kernel_size if conv_kernel_size is not None else cfg.conv_kernel_size
        )
        self.encoder_dropout = (
            encoder_dropout if encoder_dropout is not None else cfg.encoder_dropout
        )
        self.use_cln = use_cln if use_cln is not None else cfg.use_cln
        self.use_skip_connection = (
            use_skip_connection
            if use_skip_connection is not None
            else cfg.use_skip_connection
        )
        self.add_diff_step = (
            add_diff_step if add_diff_step is not None else cfg.add_diff_step
        )
        self.use_new_ffn = use_new_ffn if use_new_ffn is not None else cfg.use_new_ffn

        if enc_emb_tokens != None:
            self.use_enc_emb = True
            self.enc_emb_tokens = enc_emb_tokens
        else:
            self.use_enc_emb = False

        self.position_emb = PositionalEncoding(
            self.encoder_hidden, self.encoder_dropout
        )

        self.layers = nn.ModuleList([])
        if self.use_skip_connection:
            self.layers.extend(
                [
                    TransformerEncoderLayer(
                        self.encoder_hidden,
                        self.encoder_head,
                        self.conv_filter_size,
                        self.conv_kernel_size,
                        self.encoder_dropout,
                        self.use_cln,
                        use_skip_connection=False,
                        use_new_ffn=self.use_new_ffn,
                        add_diff_step=self.add_diff_step,
                    )
                    for i in range(
                        (self.encoder_layer + 1) // 2
                    )  # for example: 12 -> 6; 13 -> 7
                ]
            )
            self.layers.extend(
                [
                    TransformerEncoderLayer(
                        self.encoder_hidden,
                        self.encoder_head,
                        self.conv_filter_size,
                        self.conv_kernel_size,
                        self.encoder_dropout,
                        self.use_cln,
                        use_skip_connection=True,
                        use_new_ffn=self.use_new_ffn,
                        add_diff_step=self.add_diff_step,
                    )
                    for i in range(
                        self.encoder_layer - (self.encoder_layer + 1) // 2
                    )  # 12 -> 6;  13 -> 6
                ]
            )
        else:
            self.layers.extend(
                [
                    TransformerEncoderLayer(
                        self.encoder_hidden,
                        self.encoder_head,
                        self.conv_filter_size,
                        self.conv_kernel_size,
                        self.encoder_dropout,
                        self.use_cln,
                        use_new_ffn=self.use_new_ffn,
                        add_diff_step=self.add_diff_step,
                        use_skip_connection=False,
                    )
                    for i in range(self.encoder_layer)
                ]
            )

        if self.use_cln:
            self.last_ln = StyleAdaptiveLayerNorm(self.encoder_hidden)
        else:
            self.last_ln = nn.LayerNorm(self.encoder_hidden)

        if self.add_diff_step:
            self.diff_step_emb = SinusoidalPosEmb(dim=self.encoder_hidden)
            # self.diff_step_projection = nn.linear(self.encoder_hidden, self.encoder_hidden)
            # self.encoder_hidden.weight.data.normal_(0.0, 0.02)
            self.diff_step_projection = Linear2(self.encoder_hidden)

    def forward(self, x, key_padding_mask, condition=None, diffusion_step=None):
        if len(x.shape) == 2 and self.use_enc_emb:
            x = self.enc_emb_tokens(x)
            x = self.position_emb(x)
        else:
            x = self.position_emb(x)  # (B, T, d)

        if self.add_diff_step and diffusion_step != None:
            diff_step_embedding = self.diff_step_emb(diffusion_step)
            diff_step_embedding = self.diff_step_projection(diff_step_embedding)
            x = x + diff_step_embedding.unsqueeze(1)

        if self.use_skip_connection:
            skip_res_list = []
            # down
            for layer in self.layers[: self.encoder_layer // 2]:
                x = layer(x, key_padding_mask, condition)
                res = x
                skip_res_list.append(res)
            # middle
            for layer in self.layers[
                self.encoder_layer // 2 : (self.encoder_layer + 1) // 2
            ]:
                x = layer(x, key_padding_mask, condition)
            # up
            for layer in self.layers[(self.encoder_layer + 1) // 2 :]:
                skip_res = skip_res_list.pop()
                x = layer(x, key_padding_mask, condition, skip_res)
        else:
            for layer in self.layers:
                x = layer(x, key_padding_mask, condition)

        if self.use_cln:
            x = self.last_ln(x, condition)
        else:
            x = self.last_ln(x)

        return x


class DiffTransformer(nn.Module):
    def __init__(
        self,
        encoder_layer=None,
        encoder_hidden=None,
        encoder_head=None,
        conv_filter_size=None,
        conv_kernel_size=None,
        encoder_dropout=None,
        use_cln=None,
        use_skip_connection=None,
        use_new_ffn=None,
        add_diff_step=None,
        cat_diff_step=None,
        in_dim=None,
        out_dim=None,
        cond_dim=None,
        cfg=None,
    ):
        super().__init__()

        self.encoder_layer = (
            encoder_layer if encoder_layer is not None else cfg.encoder_layer
        )
        self.encoder_hidden = (
            encoder_hidden if encoder_hidden is not None else cfg.encoder_hidden
        )
        self.encoder_head = (
            encoder_head if encoder_head is not None else cfg.encoder_head
        )
        self.conv_filter_size = (
            conv_filter_size if conv_filter_size is not None else cfg.conv_filter_size
        )
        self.conv_kernel_size = (
            conv_kernel_size if conv_kernel_size is not None else cfg.conv_kernel_size
        )
        self.encoder_dropout = (
            encoder_dropout if encoder_dropout is not None else cfg.encoder_dropout
        )
        self.use_cln = use_cln if use_cln is not None else cfg.use_cln
        self.use_skip_connection = (
            use_skip_connection
            if use_skip_connection is not None
            else cfg.use_skip_connection
        )
        self.use_new_ffn = use_new_ffn if use_new_ffn is not None else cfg.use_new_ffn
        self.add_diff_step = (
            add_diff_step if add_diff_step is not None else cfg.add_diff_step
        )
        self.cat_diff_step = (
            cat_diff_step if cat_diff_step is not None else cfg.cat_diff_step
        )
        self.in_dim = in_dim if in_dim is not None else cfg.in_dim
        self.out_dim = out_dim if out_dim is not None else cfg.out_dim
        self.cond_dim = cond_dim if cond_dim is not None else cfg.cond_dim

        if self.in_dim != self.encoder_hidden:
            self.in_linear = nn.Linear(self.in_dim, self.encoder_hidden)
            self.in_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.in_dim = None

        if self.out_dim != self.encoder_hidden:
            self.out_linear = nn.Linear(self.encoder_hidden, self.out_dim)
            self.out_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.out_dim = None

        assert not ((self.cat_diff_step == True) and (self.add_diff_step == True))

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            encoder_hidden=self.encoder_hidden,
            encoder_head=self.encoder_head,
            conv_kernel_size=self.conv_kernel_size,
            conv_filter_size=self.conv_filter_size,
            encoder_dropout=self.encoder_dropout,
            use_cln=self.use_cln,
            use_skip_connection=self.use_skip_connection,
            use_new_ffn=self.use_new_ffn,
            add_diff_step=self.add_diff_step,
        )

        self.cond_project = nn.Linear(self.cond_dim, self.encoder_hidden)
        self.cond_project.weight.data.normal_(0.0, 0.02)
        self.cat_linear = nn.Linear(self.encoder_hidden * 2, self.encoder_hidden)
        self.cat_linear.weight.data.normal_(0.0, 0.02)

        if self.cat_diff_step:
            self.diff_step_emb = SinusoidalPosEmb(dim=self.encoder_hidden)
            self.diff_step_projection = Linear2(self.encoder_hidden)

    def forward(
        self,
        x,
        condition_embedding,
        key_padding_mask=None,
        reference_embedding=None,
        diffusion_step=None,
    ):
        # x: shape is (B, T, d_x)
        # key_padding_mask: shape is (B, T),  mask is 0
        # condition_embedding: from condition adapter, shape is (B, T, d_c)
        # reference_embedding: from reference encoder, shape is (B, N, d_r), or (B, 1, d_r), or (B, d_r)

        if self.in_linear != None:
            x = self.in_linear(x)
        condition_embedding = self.cond_project(condition_embedding)

        x = torch.cat([x, condition_embedding], dim=-1)
        x = self.cat_linear(x)

        if self.cat_diff_step and diffusion_step != None:
            diff_step_embedding = self.diff_step_emb(diffusion_step)
            diff_step_embedding = self.diff_step_projection(
                diff_step_embedding
            ).unsqueeze(
                1
            )  # (B, 1, d)
            x = torch.cat([diff_step_embedding, x], dim=1)
            if key_padding_mask != None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.ones(key_padding_mask.shape[0], 1).to(
                            key_padding_mask.device
                        ),
                    ],
                    dim=1,
                )

        x = self.transformer_encoder(
            x,
            key_padding_mask=key_padding_mask,
            condition=reference_embedding,
            diffusion_step=diffusion_step,
        )

        if self.cat_diff_step and diffusion_step != None:
            x = x[:, 1:, :]

        if self.out_linear != None:
            x = self.out_linear(x)

        return x


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer=None,
        encoder_hidden=None,
        encoder_head=None,
        conv_filter_size=None,
        conv_kernel_size=None,
        encoder_dropout=None,
        use_skip_connection=None,
        use_new_ffn=None,
        ref_in_dim=None,
        ref_out_dim=None,
        use_query_emb=None,
        num_query_emb=None,
        cfg=None,
    ):
        super().__init__()

        self.encoder_layer = (
            encoder_layer if encoder_layer is not None else cfg.encoder_layer
        )
        self.encoder_hidden = (
            encoder_hidden if encoder_hidden is not None else cfg.encoder_hidden
        )
        self.encoder_head = (
            encoder_head if encoder_head is not None else cfg.encoder_head
        )
        self.conv_filter_size = (
            conv_filter_size if conv_filter_size is not None else cfg.conv_filter_size
        )
        self.conv_kernel_size = (
            conv_kernel_size if conv_kernel_size is not None else cfg.conv_kernel_size
        )
        self.encoder_dropout = (
            encoder_dropout if encoder_dropout is not None else cfg.encoder_dropout
        )
        self.use_skip_connection = (
            use_skip_connection
            if use_skip_connection is not None
            else cfg.use_skip_connection
        )
        self.use_new_ffn = use_new_ffn if use_new_ffn is not None else cfg.use_new_ffn
        self.in_dim = ref_in_dim if ref_in_dim is not None else cfg.ref_in_dim
        self.out_dim = ref_out_dim if ref_out_dim is not None else cfg.ref_out_dim
        self.use_query_emb = (
            use_query_emb if use_query_emb is not None else cfg.use_query_emb
        )
        self.num_query_emb = (
            num_query_emb if num_query_emb is not None else cfg.num_query_emb
        )

        if self.in_dim != self.encoder_hidden:
            self.in_linear = nn.Linear(self.in_dim, self.encoder_hidden)
            self.in_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.in_dim = None

        if self.out_dim != self.encoder_hidden:
            self.out_linear = nn.Linear(self.encoder_hidden, self.out_dim)
            self.out_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.out_linear = None

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            encoder_hidden=self.encoder_hidden,
            encoder_head=self.encoder_head,
            conv_kernel_size=self.conv_kernel_size,
            conv_filter_size=self.conv_filter_size,
            encoder_dropout=self.encoder_dropout,
            use_new_ffn=self.use_new_ffn,
            use_cln=False,
            use_skip_connection=False,
            add_diff_step=False,
        )

        if self.use_query_emb:
            # 32 x 512
            self.query_embs = nn.Embedding(self.num_query_emb, self.encoder_hidden)
            self.query_attn = nn.MultiheadAttention(
                self.encoder_hidden, self.encoder_hidden // 64, batch_first=True
            )

    def forward(self, x_ref, key_padding_mask=None):
        # x_ref: (B, T, d_ref)
        # key_padding_mask: (B, T)
        # return speaker embedding: x_spk
        # if self.use_query_embs: shape is (B, N_query, d_out)
        # else: shape is (B, T, d_out)

        if self.in_linear != None:
            # print('x_ref:',x_ref.shape)
            x = self.in_linear(x_ref)

        x = self.transformer_encoder(
            x, key_padding_mask=key_padding_mask, condition=None, diffusion_step=None
        )  # B, T, d_out

        if self.use_query_emb:
            spk_query_emb = self.query_embs(
                torch.arange(self.num_query_emb).to(x.device)
            ).repeat(x.shape[0], 1, 1)
            # k/v b x t x d
            # q b x n x d
            spk_embs, _ = self.query_attn(
                query=spk_query_emb,
                key=x,
                value=x,
                key_padding_mask=(
                    ~(key_padding_mask.bool()) if key_padding_mask is not None else None
                ),
            )  # B, N_query, d_out
            if self.out_linear != None:
                spk_embs = self.out_linear(spk_embs)

        else:
            spk_query_emb = None
        # B x n x d
        # b x t x d
        return spk_embs, x


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()

        self.gain = Linear(cond_dim, in_dim)
        self.bias = Linear(cond_dim, in_dim)

        nn.init.xavier_uniform_(self.gain.weight)
        nn.init.constant_(self.gain.bias, 1)

        nn.init.xavier_uniform_(self.bias.weight)
        nn.init.constant_(self.bias.bias, 0)

    def forward(self, x, condition):
        gain = self.gain(condition)
        bias = self.bias(condition)
        if gain.dim() == 2:
            gain = gain.unsqueeze(-1)
        if bias.dim() == 2:
            bias = bias.unsqueeze(-1)
        return x * gain + bias


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    layer.weight.data.normal_(0.0, 0.02)
    return layer


def Linear(*args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    layer.weight.data.normal_(0.0, 0.02)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, attn_head, dilation, drop_out, has_cattn=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dilation = dilation
        self.has_cattn = has_cattn
        self.attn_head = attn_head
        self.drop_out = drop_out

        self.dilated_conv = Conv1d(
            hidden_dim, 2 * hidden_dim, 3, padding=dilation, dilation=dilation
        )
        self.diffusion_proj = Linear(hidden_dim, hidden_dim)

        self.cond_proj = Conv1d(hidden_dim, hidden_dim * 2, 1)
        self.out_proj = Conv1d(hidden_dim, hidden_dim * 2, 1)

        if self.has_cattn:
            self.attn = nn.MultiheadAttention(
                hidden_dim, attn_head, 0.1, batch_first=True
            )
            self.film = FiLM(hidden_dim * 2, hidden_dim)

            self.ln = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(self.drop_out)

    def forward(self, x, x_mask, cond, diffusion_step, spk_query_emb):
        diffusion_step = self.diffusion_proj(diffusion_step).unsqueeze(-1)  # (B, d, 1)
        cond = self.cond_proj(cond)  # (B, 2*d, T)

        y = x + diffusion_step
        if x_mask != None:
            y = y * x_mask.to(y.dtype)[:, None, :]  # (B, 2*d, T)

        if self.has_cattn:
            y_ = y.transpose(1, 2)
            y_ = self.ln(y_)

            y_, _ = self.attn(y_, spk_query_emb, spk_query_emb)  # (B, T, d)

        y = self.dilated_conv(y) + cond  # (B, 2*d, T)

        if self.has_cattn:
            y = self.film(y.transpose(1, 2), y_)  # (B, T, 2*d)
            y = y.transpose(1, 2)  # (B, 2*d, T)

        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)

        y = self.out_proj(y)

        residual, skip = torch.chunk(y, 2, dim=1)

        if x_mask != None:
            residual = residual * x_mask.to(y.dtype)[:, None, :]
            skip = skip * x_mask.to(y.dtype)[:, None, :]

        return (x + residual) / math.sqrt(2.0), skip


class DiffWaveNet(nn.Module):
    def __init__(
        self,
        cfg=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.in_dim = cfg.input_size
        self.hidden_dim = cfg.hidden_size
        self.out_dim = cfg.out_size
        self.num_layers = cfg.num_layers
        self.cross_attn_per_layer = cfg.cross_attn_per_layer
        self.dilation_cycle = cfg.dilation_cycle
        self.attn_head = cfg.attn_head
        self.drop_out = cfg.drop_out

        self.in_proj = Conv1d(self.in_dim, self.hidden_dim, 1)
        self.diffusion_embedding = SinusoidalPosEmb(self.hidden_dim)

        self.mlp = nn.Sequential(
            Linear(self.hidden_dim, self.hidden_dim * 4),
            Mish(),
            Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        self.cond_ln = nn.LayerNorm(self.hidden_dim)

        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    self.hidden_dim,
                    self.attn_head,
                    2 ** (i % self.dilation_cycle),
                    self.drop_out,
                    has_cattn=(i % self.cross_attn_per_layer == 0),
                )
                for i in range(self.num_layers)
            ]
        )

        self.skip_proj = Conv1d(self.hidden_dim, self.hidden_dim, 1)
        self.out_proj = Conv1d(self.hidden_dim, self.out_dim, 1)

        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        x,
        condition_embedding,
        key_padding_mask=None,
        reference_embedding=None,
        diffusion_step=None,
    ):
        x = x.transpose(1, 2)  # (B, T, d) -> (B, d, T)
        x_mask = key_padding_mask
        cond = condition_embedding
        spk_query_emb = reference_embedding
        diffusion_step = diffusion_step

        cond = self.cond_ln(cond)
        cond_input = cond.transpose(1, 2)

        x_input = self.in_proj(x)

        x_input = F.relu(x_input)

        diffusion_step = self.diffusion_embedding(diffusion_step).to(x.dtype)
        diffusion_step = self.mlp(diffusion_step)

        skip = []
        for _, layer in enumerate(self.layers):
            x_input, skip_connection = layer(
                x_input, x_mask, cond_input, diffusion_step, spk_query_emb
            )
            skip.append(skip_connection)

        x_input = torch.sum(torch.stack(skip), dim=0) / math.sqrt(self.num_layers)

        x_out = self.skip_proj(x_input)

        x_out = F.relu(x_out)

        x_out = self.out_proj(x_out)  # (B, 80, T)

        x_out = x_out.transpose(1, 2)

        return x_out


def override_config(base_config, new_config):
    """Update new configurations in the original dict with the new dict

    Args:
        base_config (dict): original dict to be overridden
        new_config (dict): dict with new configurations

    Returns:
        dict: updated configuration dict
    """
    for k, v in new_config.items():
        if type(v) == dict:
            if k not in base_config.keys():
                base_config[k] = {}
            base_config[k] = override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def get_lowercase_keys_config(cfg):
    """Change all keys in cfg to lower case

    Args:
        cfg (dict): dictionary that stores configurations

    Returns:
        dict: dictionary that stores configurations
    """
    updated_cfg = dict()
    for k, v in cfg.items():
        if type(v) == dict:
            v = get_lowercase_keys_config(v)
        updated_cfg[k.lower()] = v
    return updated_cfg


def save_config(save_path, cfg):
    """Save configurations into a json file

    Args:
        save_path (str): path to save configurations
        cfg (dict): dictionary that stores configurations
    """
    with open(save_path, "w") as f:
        json5.dump(
            cfg, f, ensure_ascii=False, indent=4, quote_keys=True, sort_keys=True
        )


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class Noro_VCmodel(nn.Module):
    def __init__(self, cfg, use_ref_noise=False):
        super().__init__()
        self.cfg = cfg
        self.use_ref_noise = use_ref_noise
        self.reference_encoder = ReferenceEncoder(cfg=cfg.reference_encoder)
        if cfg.diffusion.diff_model_type == "WaveNet":
            self.diffusion = Diffusion(
                cfg=cfg.diffusion,
                diff_model=DiffWaveNet(cfg=cfg.diffusion.diff_wavenet),
            )
        else:
            raise NotImplementedError()
        pitch_dim = 1
        self.content_f0_enc = nn.Sequential(
            nn.LayerNorm(
                cfg.vc_feature.content_feature_dim + pitch_dim
            ),  # 768 (for mhubert) + 1 (for f0)
            Rearrange("b t d -> b d t"),
            nn.Conv1d(
                cfg.vc_feature.content_feature_dim + pitch_dim,
                cfg.vc_feature.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            Rearrange("b d t -> b t d"),
        )

        self.reset_parameters()

    def forward(
        self,
        x=None,
        content_feature=None,
        pitch=None,
        x_ref=None,
        x_mask=None,
        x_ref_mask=None,
        noisy_x_ref=None,
    ):
        noisy_reference_embedding = None
        noisy_condition_embedding = None

        reference_embedding, encoded_x = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        # content_feature: B x T x D
        # pitch: B x T x 1
        # B x t x D+1
        # 2B x T
        condition_embedding = torch.cat([content_feature, pitch[:, :, None]], dim=-1)
        condition_embedding = self.content_f0_enc(condition_embedding)

        # 2B x T x D
        if self.use_ref_noise:
            # noisy_reference
            noisy_reference_embedding, _ = self.reference_encoder(
                x_ref=noisy_x_ref, key_padding_mask=x_ref_mask
            )
            combined_reference_embedding = (
                noisy_reference_embedding + reference_embedding
            ) / 2
        else:
            combined_reference_embedding = reference_embedding

        combined_condition_embedding = condition_embedding

        diff_out = self.diffusion(
            x=x,
            condition_embedding=combined_condition_embedding,
            x_mask=x_mask,
            reference_embedding=combined_reference_embedding,
        )
        return (
            diff_out,
            (reference_embedding, noisy_reference_embedding),
            (condition_embedding, noisy_condition_embedding),
        )

    @torch.no_grad()
    def inference(
        self,
        content_feature=None,
        pitch=None,
        x_ref=None,
        x_ref_mask=None,
        inference_steps=1000,
        sigma=1.2,
    ):
        reference_embedding, _ = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        condition_embedding = torch.cat([content_feature, pitch[:, :, None]], dim=-1)
        condition_embedding = self.content_f0_enc(condition_embedding)

        bsz, l, _ = condition_embedding.shape
        if self.cfg.diffusion.diff_model_type == "WaveNet":
            z = (
                torch.randn(bsz, l, self.cfg.diffusion.diff_wavenet.input_size).to(
                    condition_embedding.device
                )
                / sigma
            )

        x0 = self.diffusion.reverse_diffusion(
            z=z,
            condition_embedding=condition_embedding,
            x_mask=None,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
        )

        return x0

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)
