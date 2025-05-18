# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from transformers import (
    Wav2Vec2BertModel,
    SeamlessM4TFeatureExtractor,
)
from dualcodec import DualCodec
import safetensors
from huggingface_hub import hf_hub_download


def build_codec_model(cfg, device):
    dual_codec_model = DualCodec(
        sample_rate=24000,
        encoder_rates=[4, 5, 6, 8, 2],
        decoder_rates=[2, 8, 6, 5, 4],
        encoder_dim=32,
        decoder_dim=1536,
        n_codebooks=cfg.model.dual_codec.n_codebooks,
        quantizer_dropout=0.5,
        codebook_size=cfg.model.dual_codec.codebook_size,
        semantic_codebook_size=cfg.model.dual_codec.semantic_codebook_size,
        is_causal=True,
        semantic_downsample_factor=cfg.model.dual_codec.semantic_downsample_factor,
    )

    dual_codec_model.eval()
    dual_codec_model.to(device)
    safetensors.torch.load_model(
        dual_codec_model,
        hf_hub_download("amphion/dualcodec", "dualcodec_12hz_16384_4096.safetensors"),
    )
    return dual_codec_model


def build_semantic_model(cfg, w2v_bert_path, device):
    semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_path)
    semantic_model.eval()
    semantic_model.to(device)

    stat_mean_var = torch.load(
        hf_hub_download("amphion/dualcodec", "w2vbert2_mean_var_stats_emilia.pt")
    )
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    return semantic_model, semantic_mean, semantic_std


class IntsAudioTokenizer:
    def __init__(self, cfg, w2v_bert_path="facebook/w2v-bert-2.0", device="cuda"):

        self.device = device
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_path)
        self.dual_codec = build_codec_model(cfg, device)

        self.semantic_model, self.semantic_mean, self.semantic_std = (
            build_semantic_model(cfg, w2v_bert_path, device)
        )

    def __call__(
        self,
        speech: np.ndarray,
    ):
        return self.wav2token(speech)

    def wav2token(self, speech: np.ndarray):
        input_features, attention_mask = self._extract_features(speech)
        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        semantic_code = self._extract_semantic_code(input_features, attention_mask)
        return semantic_code

    def _extract_features(
        self,
        speech: np.ndarray,
    ):
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]
        return input_features, attention_mask

    def _extract_semantic_code(
        self,
        input_features,
        attention_mask,
    ):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        feat = torch.nn.functional.avg_pool1d(
            feat.transpose(1, 2),
            self.dual_codec.semantic_downsample_factor,
            self.dual_codec.semantic_downsample_factor,
        )

        semantic_code = self.dual_codec.semantic_quantize(feat)  # [B, T]

        return semantic_code
