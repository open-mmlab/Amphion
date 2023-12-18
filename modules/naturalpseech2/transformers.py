# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np


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
        x = F.relu(x)
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
    ):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.encoder_head = encoder_head
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout
        self.use_cln = use_cln

        if not self.use_cln:
            self.ln_1 = nn.LayerNorm(self.encoder_hidden)
            self.ln_2 = nn.LayerNorm(self.encoder_hidden)
        else:
            self.ln_1 = StyleAdaptiveLayerNorm(self.encoder_hidden)
            self.ln_2 = StyleAdaptiveLayerNorm(self.encoder_hidden)

        self.self_attn = nn.MultiheadAttention(
            self.encoder_hidden, self.encoder_head, batch_first=True
        )

        self.ffn = TransformerFFNLayer(
            self.encoder_hidden,
            self.conv_filter_size,
            self.conv_kernel_size,
            self.encoder_dropout,
        )

    def forward(self, x, key_padding_mask, conditon=None):
        # x: (B, T, d); key_padding_mask: (B, T), mask is 0; condition: (B, T, d)

        # self attention
        residual = x
        if self.use_cln:
            x = self.ln_1(x, conditon)
        else:
            x = self.ln_1(x)

        if key_padding_mask != None:
            key_padding_mask_input = ~(key_padding_mask.bool())
        else:
            key_padding_mask_input = None
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask_input
        )
        x = F.dropout(x, self.encoder_dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        if self.use_cln:
            x = self.ln_2(x, conditon)
        else:
            x = self.ln_2(x)
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

        if enc_emb_tokens != None:
            self.use_enc_emb = True
            self.enc_emb_tokens = enc_emb_tokens
        else:
            self.use_enc_emb = False

        self.position_emb = PositionalEncoding(
            self.encoder_hidden, self.encoder_dropout
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    self.encoder_hidden,
                    self.encoder_head,
                    self.conv_filter_size,
                    self.conv_kernel_size,
                    self.encoder_dropout,
                    self.use_cln,
                )
                for i in range(self.encoder_layer)
            ]
        )

        if self.use_cln:
            self.last_ln = StyleAdaptiveLayerNorm(self.encoder_hidden)
        else:
            self.last_ln = nn.LayerNorm(self.encoder_hidden)

    def forward(self, x, key_padding_mask, condition=None):
        if len(x.shape) == 2 and self.use_enc_emb:
            x = self.enc_emb_tokens(x)
            x = self.position_emb(x)
        else:
            x = self.position_emb(x)  # (B, T, d)

        for layer in self.layers:
            x = layer(x, key_padding_mask, condition)

        if self.use_cln:
            x = self.last_ln(x, condition)
        else:
            x = self.last_ln(x)

        return x


class DurationPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_size = cfg.input_size
        self.filter_size = cfg.filter_size
        self.kernel_size = cfg.kernel_size
        self.conv_layers = cfg.conv_layers
        self.cross_attn_per_layer = cfg.cross_attn_per_layer
        self.attn_head = cfg.attn_head
        self.drop_out = cfg.drop_out

        self.conv = nn.ModuleList()
        self.cattn = nn.ModuleList()

        for idx in range(self.conv_layers):
            in_dim = self.input_size if idx == 0 else self.filter_size
            self.conv += [
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        self.filter_size,
                        self.kernel_size,
                        padding=self.kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(self.filter_size),
                    nn.Dropout(self.drop_out),
                )
            ]
            if idx % self.cross_attn_per_layer == 0:
                self.cattn.append(
                    torch.nn.Sequential(
                        nn.MultiheadAttention(
                            self.filter_size,
                            self.attn_head,
                            batch_first=True,
                            kdim=self.filter_size,
                            vdim=self.filter_size,
                        ),
                        nn.LayerNorm(self.filter_size),
                        nn.Dropout(0.2),
                    )
                )

        self.linear = nn.Linear(self.filter_size, 1)
        self.linear.weight.data.normal_(0.0, 0.02)

    def forward(self, x, mask, ref_emb, ref_mask):
        """
        input:
        x: (B, N, d)
        mask: (B, N), mask is 0
        ref_emb: (B, d, T')
        ref_mask: (B, T'), mask is 0

        output:
        dur_pred: (B, N)
        dur_pred_log: (B, N)
        dur_pred_round: (B, N)
        """

        input_ref_mask = ~(ref_mask.bool())  # (B, T')
        # print(input_ref_mask)

        x = x.transpose(1, -1)  # (B, N, d) -> (B, d, N)

        for idx, (conv, act, ln, dropout) in enumerate(self.conv):
            res = x
            # print(torch.min(x), torch.max(x))
            if idx % self.cross_attn_per_layer == 0:
                attn_idx = idx // self.cross_attn_per_layer
                attn, attn_ln, attn_drop = self.cattn[attn_idx]

                attn_res = y_ = x.transpose(1, 2)  # (B, d, N) -> (B, N, d)

                y_ = attn_ln(y_)
                # print(torch.min(y_), torch.min(y_))
                # print(torch.min(ref_emb), torch.max(ref_emb))
                y_, _ = attn(
                    y_,
                    ref_emb.transpose(1, 2),
                    ref_emb.transpose(1, 2),
                    key_padding_mask=input_ref_mask,
                )
                # y_, _ = attn(y_, ref_emb.transpose(1, 2), ref_emb.transpose(1, 2))
                # print(torch.min(y_), torch.min(y_))
                y_ = attn_drop(y_)
                y_ = (y_ + attn_res) / math.sqrt(2.0)

                x = y_.transpose(1, 2)

            x = conv(x)
            # print(torch.min(x), torch.max(x))
            x = act(x)
            x = ln(x.transpose(1, 2))
            # print(torch.min(x), torch.max(x))
            x = x.transpose(1, 2)

            x = dropout(x)

            if idx != 0:
                x += res

            if mask is not None:
                x = x * mask.to(x.dtype)[:, None, :]

        x = self.linear(x.transpose(1, 2))
        x = torch.squeeze(x, -1)

        dur_pred = x.exp() - 1
        dur_pred_round = torch.clamp(torch.round(x.exp() - 1), min=0).long()

        return {
            "dur_pred_log": x,
            "dur_pred": dur_pred,
            "dur_pred_round": dur_pred_round,
        }


class PitchPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_size = cfg.input_size
        self.filter_size = cfg.filter_size
        self.kernel_size = cfg.kernel_size
        self.conv_layers = cfg.conv_layers
        self.cross_attn_per_layer = cfg.cross_attn_per_layer
        self.attn_head = cfg.attn_head
        self.drop_out = cfg.drop_out

        self.conv = nn.ModuleList()
        self.cattn = nn.ModuleList()

        for idx in range(self.conv_layers):
            in_dim = self.input_size if idx == 0 else self.filter_size
            self.conv += [
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        self.filter_size,
                        self.kernel_size,
                        padding=self.kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(self.filter_size),
                    nn.Dropout(self.drop_out),
                )
            ]
            if idx % self.cross_attn_per_layer == 0:
                self.cattn.append(
                    torch.nn.Sequential(
                        nn.MultiheadAttention(
                            self.filter_size,
                            self.attn_head,
                            batch_first=True,
                            kdim=self.filter_size,
                            vdim=self.filter_size,
                        ),
                        nn.LayerNorm(self.filter_size),
                        nn.Dropout(0.2),
                    )
                )

        self.linear = nn.Linear(self.filter_size, 1)
        self.linear.weight.data.normal_(0.0, 0.02)

    def forward(self, x, mask, ref_emb, ref_mask):
        """
        input:
        x: (B, N, d)
        mask: (B, N), mask is 0
        ref_emb: (B, d, T')
        ref_mask: (B, T'), mask is 0

        output:
        pitch_pred: (B, T)
        """

        input_ref_mask = ~(ref_mask.bool())  # (B, T')

        x = x.transpose(1, -1)  # (B, N, d) -> (B, d, N)

        for idx, (conv, act, ln, dropout) in enumerate(self.conv):
            res = x
            if idx % self.cross_attn_per_layer == 0:
                attn_idx = idx // self.cross_attn_per_layer
                attn, attn_ln, attn_drop = self.cattn[attn_idx]

                attn_res = y_ = x.transpose(1, 2)  # (B, d, N) -> (B, N, d)

                y_ = attn_ln(y_)
                y_, _ = attn(
                    y_,
                    ref_emb.transpose(1, 2),
                    ref_emb.transpose(1, 2),
                    key_padding_mask=input_ref_mask,
                )
                # y_, _ = attn(y_, ref_emb.transpose(1, 2), ref_emb.transpose(1, 2))
                y_ = attn_drop(y_)
                y_ = (y_ + attn_res) / math.sqrt(2.0)

                x = y_.transpose(1, 2)

            x = conv(x)
            x = act(x)
            x = ln(x.transpose(1, 2))
            x = x.transpose(1, 2)

            x = dropout(x)

            if idx != 0:
                x += res

        x = self.linear(x.transpose(1, 2))
        x = torch.squeeze(x, -1)

        return x


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


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        device = x.device
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
