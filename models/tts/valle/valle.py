# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from https://github.com/lifeiteng/vall-e/blob/main/valle/models/valle.py

import random
from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from utils.util import make_pad_mask
from utils.topk_sampling import topk_sampling
from modules.general import Transpose
from modules.encoder import TokenEmbedding
from modules.general import PromptedFeatures
from modules.transformer import SinePositionalEmbedding
from modules.norms import AdaptiveLayerNorm, LayerNorm
from modules.transformer.transformer import TransformerEncoder, TransformerEncoderLayer


class VALLE(nn.Module):
    def __init__(
        self,
        cfg,
        decoder_cls=TransformerEncoder,
        decoder_layer_cls=TransformerEncoderLayer,
    ):
        super().__init__()
        decoder_dim = cfg.decoder_dim
        nhead = cfg.nhead
        nar_scale_factor = cfg.nar_scale_factor
        num_quantizers = cfg.num_quantizers
        num_decoder_layers = cfg.num_decoder_layers
        nar_decoder_dim = int(decoder_dim * nar_scale_factor)

        self.ar_text_embedding = TokenEmbedding(decoder_dim, cfg.text_token_num)
        self.nar_text_embedding = TokenEmbedding(nar_decoder_dim, cfg.text_token_num)

        self.ar_audio_prepend_bos = cfg.prepend_bos
        self.ar_audio_embedding = TokenEmbedding(
            decoder_dim, cfg.audio_token_num + 1 + int(cfg.prepend_bos)
        )
        self.audio_token_num = cfg.audio_token_num

        # PreNet of AR
        if cfg.add_prenet:
            self.ar_text_prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(decoder_dim, decoder_dim, kernel_size=5, padding="same"),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(decoder_dim, decoder_dim, kernel_size=5, padding="same"),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(decoder_dim, decoder_dim, kernel_size=5, padding="same"),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                Transpose(),
                nn.Linear(decoder_dim, decoder_dim),
            )

            self.ar_audio_prenet = nn.Sequential(
                nn.Linear(decoder_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, decoder_dim),
            )
        else:
            self.ar_text_prenet = nn.Identity()
            self.ar_audio_prenet = nn.Identity()

        self.ar_text_position = SinePositionalEmbedding(
            decoder_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            decoder_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        self.ar_decoder = decoder_cls(
            decoder_layer_cls(
                decoder_dim,
                nhead,
                dim_feedforward=decoder_dim * 4,  # *4?
                dropout=0.1,
                batch_first=True,
                norm_first=cfg.norm_first,
            ),
            num_layers=num_decoder_layers,
            norm=LayerNorm(decoder_dim) if cfg.norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(
            decoder_dim, cfg.audio_token_num + 1, bias=False
        )

        self.ar_accuracy_metric = MulticlassAccuracy(
            cfg.audio_token_num + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=cfg.audio_token_num,
        )

        self.rng = random.Random(0)
        self.num_heads = nhead
        self.prefix_mode = cfg.prefix_mode
        self.num_quantizers = num_quantizers

        assert num_quantizers >= 1
        if num_quantizers > 1:
            self.nar_audio_embeddings = nn.ModuleList(
                [
                    TokenEmbedding(nar_decoder_dim, cfg.audio_token_num + 1)
                ]  # Why the first layer is audio_token_num + 1?
                + [
                    TokenEmbedding(nar_decoder_dim, cfg.audio_token_num)
                    for i in range(num_quantizers - 1)
                ]
            )

            if cfg.add_prenet:
                self.nar_text_prenet = nn.Sequential(
                    Transpose(),
                    nn.Conv1d(
                        nar_decoder_dim, nar_decoder_dim, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_decoder_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_decoder_dim, nar_decoder_dim, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_decoder_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_decoder_dim, nar_decoder_dim, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_decoder_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    Transpose(),
                    nn.Linear(nar_decoder_dim, nar_decoder_dim),
                )
                self.nar_audio_prenet = nn.Sequential(
                    nn.Linear(nar_decoder_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, nar_decoder_dim),
                )
            else:
                self.nar_text_prenet = nn.Identity()
                self.nar_audio_prenet = nn.Identity()

            self.nar_text_position = SinePositionalEmbedding(
                nar_decoder_dim,
                dropout=0.0,
                scale=False,
                alpha=False,
            )
            self.nar_audio_position = SinePositionalEmbedding(
                nar_decoder_dim,
                dropout=0.1,
                scale=False,
                alpha=False,
            )

            self.nar_decoder = decoder_cls(
                decoder_layer_cls(
                    nar_decoder_dim,
                    int(nhead * nar_scale_factor),
                    dim_feedforward=nar_decoder_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=cfg.norm_first,
                    adaptive_layer_norm=True,
                ),
                num_layers=int(num_decoder_layers * nar_scale_factor),
                norm=(
                    AdaptiveLayerNorm(
                        nar_decoder_dim, norm=nn.LayerNorm(nar_decoder_dim)
                    )
                    if cfg.norm_first
                    else None
                ),
            )
            self.nar_predict_layers = nn.ModuleList(
                [
                    nn.Linear(nar_decoder_dim, cfg.audio_token_num, bias=False)
                    for i in range(num_quantizers - 1)
                ]
            )
            self.nar_stage_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_decoder_dim, 1) for i in range(num_quantizers - 1)]
            )

            if cfg.share_embedding:
                for j in range(0, num_quantizers - 2):
                    self.nar_predict_layers[j].weight = self.nar_audio_embeddings[
                        j + 2
                    ].weight

            self.nar_accuracy_metric = MulticlassAccuracy(
                cfg.audio_token_num + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=cfg.audio_token_num,
            )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        reduction: str = "sum",
        train_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        y_prompts_codes = None
        if isinstance(y, PromptedFeatures):
            y_prompts_codes, y = y.data
            prompts_len, y_lens = y_lens.data
            assert prompts_len.min() == prompts_len.max()
            assert self.prefix_mode == 4
            y_prompts_codes = y_prompts_codes.type(torch.int64)

        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        x_mask = make_pad_mask(x_lens).to(x.device)
        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        text = x
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        y, targets = self.pad_y_eos(
            codes[..., 0], y_mask_int, eos_id=self.audio_token_num
        )
        self.y_mask_int = y_mask_int

        metrics = {}
        total_loss = 0.0

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        if self.ar_audio_prepend_bos:
            ar_xy_padding_mask = torch.concat(
                [x_mask, F.pad(y_mask, (1, 0), value=False)], dim=1
            )
        else:
            ar_xy_padding_mask = xy_padding_mask
        self.xy_padding_mask = xy_padding_mask
        self.ar_xy_padding_mask = ar_xy_padding_mask

        # AR Decoder
        if train_stage in [0, 1]:
            ar_loss, ar_metrics = self._forward_ar_decoder(
                text, x_lens.max(), y, y_lens.max(), targets, x_mask, y_mask, reduction
            )
            total_loss += ar_loss
            metrics["AR_Top100Acc"] = ar_metrics

        # NAR Decoder
        if self.ar_audio_prepend_bos:
            y = y[:, 1:]

        if self.num_quantizers > 1 and train_stage in [0, 2]:
            nar_loss, nar_metrics = self._forward_nar_decoder(
                text,
                x_lens,
                y,
                y_lens,
                codes,
                y_prompts_codes,
                x_mask,
                y_mask,
                reduction,
            )
            total_loss += nar_loss
            metrics["NAR_Top100Acc"] = nar_metrics

        if train_stage == 0:
            total_loss = total_loss / 2.0

        return total_loss, metrics

    def _forward_ar_decoder(
        self, x, x_len, y, y_lens, targets, x_mask, y_mask, reduction
    ):
        x = self.ar_text_embedding(x)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        y_len = y_lens.max() + int(self.ar_audio_prepend_bos)

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            self.ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(bsz * self.num_heads, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)

        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask

        y_emb = self.ar_audio_embedding(y)
        y_emb = self.ar_audio_prenet(y_emb)
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = torch.concat([x, y_pos], dim=1)

        xy_dec, _ = self.ar_decoder(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
        ar_loss = F.cross_entropy(logits, targets, reduction=reduction)

        ar_metrics = self.ar_accuracy_metric(
            logits.detach(), targets
        ).item() * y_lens.sum().type(torch.float32)

        return ar_loss, ar_metrics

    def _forward_nar_decoder(
        self, x, x_lens, y, y_lens, codes, y_prompts_codes, x_mask, y_mask, reduction
    ):
        num_nar_layers = self.num_quantizers - 1
        nar_stage = self.rng.choices(
            [_k for _k in range(1, self.num_quantizers)],
            weights=[1.0 / num_nar_layers] * num_nar_layers,
            k=1,
        )[0]

        x = self.nar_text_embedding(x)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        y_emb, prefix_len = self._prepare_prompts(
            y, y_lens, codes, nar_stage, y_prompts_codes
        )

        y_len = y_lens.max()
        targets = codes[..., nar_stage] + self.audio_token_num * self.y_mask_int
        if self.prefix_mode in [2, 4]:
            xy_padding_mask = torch.concat(
                [
                    x_mask,
                    F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False),
                ],
                dim=1,
            )
        elif self.prefix_mode == 1:
            targets = targets[:, prefix_len:]

        y_pos = self.nar_audio_prenet(y_emb)
        y_pos = self.nar_audio_position(y_pos)
        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.nar_decoder(
            (xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
            src_key_padding_mask=self.xy_padding_mask,
        )
        xy_dec = xy_dec[:, x_lens.max() + prefix_len :]
        if self.prefix_mode == 4:
            prefix_len = 0
        logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(0, 2, 1)

        total_length = (y_lens).sum().type(torch.float32)
        nar_loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.audio_token_num,
            reduction=reduction,
        ) * (total_length / (total_length - prefix_len * x.shape[0]))
        nar_metrics = (
            self.nar_accuracy_metric(
                F.pad(
                    logits.detach(),
                    (0, 0, 0, 1, 0, 0),
                    value=logits.min().cpu().item(),
                ),
                targets,
            ).item()
            * total_length
        )
        return nar_loss, nar_metrics

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()
        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=self.audio_token_num + 1)

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        while True:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)

            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(
                y.device
            )

            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if (
                torch.argmax(logits, dim=-1)[0] == self.audio_token_num
                or samples[0, 0] == self.audio_token_num
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError("well trained model shouldn't reach here.")

                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)

        # Non-AR Decoders
        y_emb = self.nar_audio_embeddings[0](y[:, int(self.ar_audio_prepend_bos) :])

        if self.prefix_mode in [2, 4]:
            enrolled_len = enroll_x_lens.max().item()
            # SOS + Synthesis Text + EOS
            text = torch.concat(
                [
                    text[:, :1],
                    text[:, enrolled_len - 1 :],
                ],
                dim=1,
            )
            text_len = text_len - (enrolled_len - 2)
            assert text.shape[0] == 1

        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, :prefix_len] += embedding_layer(prompts[..., i + 1])
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](prompts[..., j])

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

    def continual(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)
        assert self.num_quantizers == 8

        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()

        prefix_len = min(int(y.shape[1] * 0.5), 3 * 75)

        # AR Decoder
        prompts = y[:, :prefix_len]

        codes = [y[:, prefix_len:, 0]]
        # Non-AR Decoders
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        y_emb = self.nar_audio_embeddings[0](y[..., 0])

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_position(y_emb)
                y_pos = self.nar_audio_prenet(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, :prefix_len] += embedding_layer(prompts[..., i + 1])
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, 8):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](prompts[..., j])

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)

    def stage_parameters(self, stage: int = 1) -> Iterator[nn.Parameter]:
        assert stage > 0
        if stage == 1:
            for name, param in self.named_parameters():
                if name.startswith("ar_"):
                    yield param

        if stage == 2:
            for name, param in self.named_parameters():
                if name.startswith("nar_"):
                    yield param

    def stage_named_parameters(
        self, stage: int = 1
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith("ar_"):
                    yield pair

        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith("nar_"):
                    yield pair

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        if self.ar_audio_prepend_bos:
            return (
                F.pad(targets[:, :-1], (1, 0), value=self.audio_token_num + 1),
                targets,
            )

        return targets[:, :-1], targets[:, 1:]

    def _prepare_prompts(self, y, y_lens, codes, nar_stage, y_prompts_codes):
        # 5.1 For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds
        # from the same utterance.
        # We implement this differently.
        if self.prefix_mode == 0:
            # no prefix
            prefix_len = 0
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, nar_stage):
                # Formula (4) (5)
                y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
        elif self.prefix_mode == 1:
            # prefix at begining
            int_low = (0.25 * y_lens.min()).type(torch.int64).item()
            prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
            prefix_len = min(prefix_len, 225)  # 24000/320 * 3s = 225 frames

            y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])
            y_emb = self.nar_audio_embeddings[0](y[:, prefix_len:])
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](codes[:, :prefix_len, j])
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[:, prefix_len:, j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        elif self.prefix_mode in [2, 4]:
            if self.prefix_mode == 2:
                # random prefix
                prefix_len = min(225, int(0.25 * y_lens.min().item()))

                y_prompts_codes = []
                for b in range(codes.shape[0]):
                    start = self.rng.randint(0, y_lens[b].item() - prefix_len)
                    y_prompts_codes.append(
                        torch.clone(codes[b, start : start + prefix_len])
                    )
                    codes[b, start : start + prefix_len, nar_stage] = NUM_AUDIO_TOKENS
                y_prompts_codes = torch.stack(y_prompts_codes, dim=0)
            else:
                prefix_len = y_prompts_codes.shape[1]

            y_prompts = self.nar_audio_embeddings[0](y_prompts_codes[..., 0])
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](y_prompts_codes[..., j])
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[..., j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        else:
            raise ValueError

        return y_emb, prefix_len
