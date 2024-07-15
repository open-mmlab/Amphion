# Copyright (c) 2024 Amphion.
#
# This code is modified from https://github.com/imdanboy/jets/blob/main/espnet2/gan_tts/jets/generator.py
# Licensed under Apache License 2.0

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.transformer.Models import Encoder, Decoder
from modules.transformer.Layers import PostNet
from collections import OrderedDict
from models.tts.jets.alignments import (
    AlignmentModule,
    viterbi_decode,
    average_by_duration,
    make_pad_mask,
    make_non_pad_mask,
    get_random_segments,
)
from models.tts.jets.length_regulator import GaussianUpsampling
from models.vocoders.gan.generator.hifigan import HiFiGAN
import os
import json

from utils.util import load_config


def get_mask_from_lengths(lengths, max_len=None):
    device = lengths.device
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


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


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, cfg):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(cfg)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(cfg)
        self.energy_predictor = VariancePredictor(cfg)

        # assign the pitch/energy feature level
        if cfg.preprocess.use_frame_pitch:
            self.pitch_feature_level = "frame_level"
            self.pitch_dir = cfg.preprocess.pitch_dir
        else:
            self.pitch_feature_level = "phoneme_level"
            self.pitch_dir = cfg.preprocess.phone_pitch_dir

        if cfg.preprocess.use_frame_energy:
            self.energy_feature_level = "frame_level"
            self.energy_dir = cfg.preprocess.energy_dir
        else:
            self.energy_feature_level = "phoneme_level"
            self.energy_dir = cfg.preprocess.phone_energy_dir

        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = cfg.model.variance_embedding.pitch_quantization
        energy_quantization = cfg.model.variance_embedding.energy_quantization
        n_bins = cfg.model.variance_embedding.n_bins
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        with open(
            os.path.join(
                cfg.preprocess.processed_dir,
                cfg.dataset[0],
                self.energy_dir,
                "statistics.json",
            )
        ) as f:
            stats = json.load(f)
            stats = stats[cfg.dataset[0] + "_" + cfg.dataset[0]]
            mean, std = (
                stats["voiced_positions"]["mean"],
                stats["voiced_positions"]["std"],
            )
            energy_min = (stats["total_positions"]["min"] - mean) / std
            energy_max = (stats["total_positions"]["max"] - mean) / std

        with open(
            os.path.join(
                cfg.preprocess.processed_dir,
                cfg.dataset[0],
                self.pitch_dir,
                "statistics.json",
            )
        ) as f:
            stats = json.load(f)
            stats = stats[cfg.dataset[0] + "_" + cfg.dataset[0]]
            mean, std = (
                stats["voiced_positions"]["mean"],
                stats["voiced_positions"]["std"],
            )
            pitch_min = (stats["total_positions"]["min"] - mean) / std
            pitch_max = (stats["total_positions"]["max"] - mean) / std

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, cfg.model.transformer.encoder_hidden
        )
        self.energy_embedding = nn.Embedding(
            n_bins, cfg.model.transformer.encoder_hidden
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        pitch_embedding=None,
        energy_embedding=None,
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)

        x = x + pitch_embedding
        x = x + energy_embedding

        pitch_prediction = self.pitch_predictor(x, src_mask)
        energy_prediction = self.energy_predictor(x, src_mask)

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )

    def inference(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        pitch_embedding=None,
        energy_embedding=None,
    ):

        p_outs = self.pitch_predictor(x, src_mask)
        e_outs = self.energy_predictor(x, src_mask)
        d_outs = self.duration_predictor(x, src_mask)

        return p_outs, e_outs, d_outs


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


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, cfg):
        super(VariancePredictor, self).__init__()

        self.input_size = cfg.model.transformer.encoder_hidden
        self.filter_size = cfg.model.variance_predictor.filter_size
        self.kernel = cfg.model.variance_predictor.kernel_size
        self.conv_output_size = cfg.model.variance_predictor.filter_size
        self.dropout = cfg.model.variance_predictor.dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Jets(nn.Module):
    def __init__(self, cfg) -> None:
        super(Jets, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.model)
        self.variance_adaptor = VarianceAdaptor(cfg)
        self.decoder = Decoder(cfg.model)
        self.length_regulator_infer = LengthRegulator()
        self.mel_linear = nn.Linear(
            cfg.model.transformer.decoder_hidden,
            cfg.preprocess.n_mel,
        )
        self.postnet = PostNet(n_mel_channels=cfg.preprocess.n_mel)

        self.speaker_emb = None
        if cfg.train.multi_speaker_training:
            with open(
                os.path.join(
                    cfg.preprocess.processed_dir, cfg.dataset[0], "spk2id.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                cfg.model.transformer.encoder_hidden,
            )

        output_dim = cfg.preprocess.n_mel
        attention_dim = 256
        self.alignment_module = AlignmentModule(attention_dim, output_dim)

        # NOTE(kan-bayashi): We use continuous pitch + FastPitch style avg
        pitch_embed_kernel_size: int = 9
        pitch_embed_dropout: float = 0.5
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=attention_dim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(pitch_embed_dropout),
        )

        # NOTE(kan-bayashi): We use continuous enegy + FastPitch style avg
        energy_embed_kernel_size: int = 9
        energy_embed_dropout: float = 0.5
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=attention_dim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(energy_embed_dropout),
        )

        # define length regulator
        self.length_regulator = GaussianUpsampling()

        self.segment_size = cfg.train.segment_size

        # Define HiFiGAN generator
        hifi_cfg = load_config("egs/vocoder/gan/hifigan/exp_config.json")
        # hifi_cfg.model.hifigan.resblock_kernel_sizes = [3, 7, 11]
        hifi_cfg.preprocess.n_mel = attention_dim
        self.generator = HiFiGAN(hifi_cfg)

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def forward(self, data, p_control=1.0, e_control=1.0, d_control=1.0):
        speakers = data["spk_id"]
        texts = data["texts"]
        src_lens = data["text_len"]
        max_src_len = max(src_lens)
        feats = data["mel"]
        mel_lens = data["target_len"] if "target_len" in data else None
        feats_lengths = mel_lens
        max_mel_len = max(mel_lens) if "target_len" in data else None
        p_targets = data["pitch"] if "pitch" in data else None
        e_targets = data["energy"] if "energy" in data else None
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Forward alignment module and obtain duration, averaged pitch, energy
        h_masks = make_pad_mask(src_lens).to(output.device)
        log_p_attn = self.alignment_module(
            output, feats, src_lens, feats_lengths, h_masks
        )
        ds, bin_loss = viterbi_decode(log_p_attn, src_lens, feats_lengths)
        ps = average_by_duration(
            ds, p_targets.squeeze(-1), src_lens, feats_lengths
        ).unsqueeze(-1)
        es = average_by_duration(
            ds, e_targets.squeeze(-1), src_lens, feats_lengths
        ).unsqueeze(-1)
        p_embs = self.pitch_embed(ps.transpose(1, 2)).transpose(1, 2)
        e_embs = self.energy_embed(es.transpose(1, 2)).transpose(1, 2)

        # FastSpeech2 variance adaptor
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            ds,
            p_control,
            e_control,
            d_control,
            ps,
            es,
        )

        # forward decoder
        zs, _ = self.decoder(output, mel_masks)  # (B, T_feats, adim)

        # get random segments
        z_segments, z_start_idxs = get_random_segments(
            zs.transpose(1, 2),
            feats_lengths,
            self.segment_size,
        )

        # forward generator
        wav = self.generator(z_segments)

        return (
            wav,
            bin_loss,
            log_p_attn,
            z_start_idxs,
            log_d_predictions,
            ds,
            p_predictions,
            ps,
            e_predictions,
            es,
            src_lens,
            feats_lengths,
        )

    def inference(self, data, p_control=1.0, e_control=1.0, d_control=1.0):
        speakers = data["spk_id"]
        texts = data["texts"]
        src_lens = data["text_len"]
        max_src_len = max(src_lens)
        mel_lens = data["target_len"] if "target_len" in data else None
        feats_lengths = mel_lens
        max_mel_len = max(mel_lens) if "target_len" in data else None
        p_targets = data["pitch"] if "pitch" in data else None
        e_targets = data["energy"] if "energy" in data else None
        d_targets = data["durations"] if "durations" in data else None
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        x_masks = self._source_mask(src_lens)
        hs = self.encoder(texts, src_masks)

        (
            p_outs,
            e_outs,
            d_outs,
        ) = self.variance_adaptor.inference(
            hs,
            src_masks,
        )

        p_embs = self.pitch_embed(p_outs.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        e_embs = self.energy_embed(e_outs.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        hs = hs + e_embs + p_embs

        # Duration predictor inference mode: log_d_pred to d_pred
        offset = 1.0
        d_predictions = torch.clamp(
            torch.round(d_outs.exp() - offset), min=0
        ).long()  # avoid negative value

        # forward decoder
        hs, mel_len = self.length_regulator_infer(hs, d_predictions, max_mel_len)
        mel_mask = get_mask_from_lengths(mel_len)
        zs, _ = self.decoder(hs, mel_mask)  # (B, T_feats, adim)

        # forward generator
        wav = self.generator(zs.transpose(1, 2))

        return wav, d_predictions
