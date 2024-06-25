# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#  This code is modified from https://github.com/ming024/FastSpeech2/blob/master/model/fastspeech2.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.transformer.Models import Encoder, Decoder
from modules.transformer.Layers import PostNet
from collections import OrderedDict

import os
import json


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
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding

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

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


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


class FastSpeech2(nn.Module):
    def __init__(self, cfg) -> None:
        super(FastSpeech2, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.model)
        self.variance_adaptor = VarianceAdaptor(cfg)
        self.decoder = Decoder(cfg.model)
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

    def forward(self, data, p_control=1.0, e_control=1.0, d_control=1.0):
        speakers = data["spk_id"]
        texts = data["texts"]
        src_lens = data["text_len"]
        max_src_len = max(src_lens)
        mel_lens = data["target_len"] if "target_len" in data else None
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

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

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
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return {
            "output": output,
            "postnet_output": postnet_output,
            "p_predictions": p_predictions,
            "e_predictions": e_predictions,
            "log_d_predictions": log_d_predictions,
            "d_rounded": d_rounded,
            "src_masks": src_masks,
            "mel_masks": mel_masks,
            "src_lens": src_lens,
            "mel_lens": mel_lens,
        }


class FastSpeech2Loss(nn.Module):
    """FastSpeech2 Loss"""

    def __init__(self, cfg):
        super(FastSpeech2Loss, self).__init__()
        if cfg.preprocess.use_frame_pitch:
            self.pitch_feature_level = "frame_level"
        else:
            self.pitch_feature_level = "phoneme_level"

        if cfg.preprocess.use_frame_energy:
            self.energy_feature_level = "frame_level"
        else:
            self.energy_feature_level = "phoneme_level"

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, data, predictions):
        mel_targets = data["mel"]
        pitch_targets = data["pitch"].float()
        energy_targets = data["energy"].float()
        duration_targets = data["durations"]

        mel_predictions = predictions["output"]
        postnet_mel_predictions = predictions["postnet_output"]
        pitch_predictions = predictions["p_predictions"]
        energy_predictions = predictions["e_predictions"]
        log_duration_predictions = predictions["log_d_predictions"]
        src_masks = predictions["src_masks"]
        mel_masks = predictions["mel_masks"]

        src_masks = ~src_masks
        mel_masks = ~mel_masks

        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, : mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return {
            "loss": total_loss,
            "mel_loss": mel_loss,
            "postnet_mel_loss": postnet_mel_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
            "duration_loss": duration_loss,
        }
