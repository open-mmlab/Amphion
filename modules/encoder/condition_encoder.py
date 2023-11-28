# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torchaudio.models import Conformer
from models.svc.transformer.transformer import PositionalEncoding

from utils.f0 import f0_to_coarse


class ContentEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # if input_dim != 0:
        #     self.nn = nn.Linear(input_dim, output_dim)

        assert input_dim != 0

        # TODO: introduce conformer
        self.pos_encoder = PositionalEncoding(input_dim)
        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=2,
            ffn_dim=256,
            num_layers=6,
            depthwise_conv_kernel_size=3,
        )
        self.nn = nn.Linear(input_dim, output_dim)

    def forward(self, x, length=None):
        # x: (N, seq_len, input_dim) -> (N, seq_len, output_dim)
        x = self.pos_encoder(x)
        x, _ = self.conformer(x, length)
        return self.nn(x)


class MelodyEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = self.cfg.input_melody_dim
        self.output_dim = self.cfg.output_melody_dim
        self.n_bins = self.cfg.n_bins_melody
        self.pitch_min = self.cfg.pitch_min
        self.pitch_max = self.cfg.pitch_max

        if self.input_dim != 0:
            if self.n_bins == 0:
                # Not use quantization
                self.nn = nn.Linear(self.input_dim, self.output_dim)
            else:
                self.f0_min = cfg.f0_min
                self.f0_max = cfg.f0_max

                self.nn = nn.Embedding(
                    num_embeddings=self.n_bins,
                    embedding_dim=self.output_dim,
                    padding_idx=None,
                )
                self.uv_embedding = nn.Embedding(2, self.output_dim)
                # self.conformer = Conformer(
                #     input_dim=self.output_dim,
                #     num_heads=4,
                #     ffn_dim=128,
                #     num_layers=4,
                #     depthwise_conv_kernel_size=3,
                # )

    def forward(self, x, uv=None, length=None):
        # x: (N, frame_len)
        # print(x.shape)
        if self.n_bins == 0:
            x = x.unsqueeze(-1)
        else:
            x = f0_to_coarse(x, self.n_bins, self.f0_min, self.f0_max)
            x = self.nn(x)
            if uv is not None:
                uv = self.uv_embedding(uv)
                x = x + uv
            # x, _ = self.conformer(x, length)
        return x


class LoudnessEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = self.cfg.input_loudness_dim
        self.output_dim = self.cfg.output_loudness_dim
        self.n_bins = self.cfg.n_bins_loudness

        if self.input_dim != 0:
            if self.n_bins == 0:
                # Not use quantization
                self.nn = nn.Linear(self.input_dim, self.output_dim)
            else:
                # TODO: set trivially now
                self.loudness_min = 1e-30
                self.loudness_max = 1.5

                if cfg.use_log_loudness:
                    self.energy_bins = nn.Parameter(
                        torch.exp(
                            torch.linspace(
                                np.log(self.loudness_min),
                                np.log(self.loudness_max),
                                self.n_bins - 1,
                            )
                        ),
                        requires_grad=False,
                    )

                self.nn = nn.Embedding(
                    num_embeddings=self.n_bins,
                    embedding_dim=self.output_dim,
                    padding_idx=None,
                )

    def forward(self, x):
        # x: (N, frame_len)
        if self.n_bins == 0:
            x = x.unsqueeze(-1)
        else:
            x = torch.bucketize(x, self.energy_bins)
        return self.nn(x)


class SingerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = 1
        self.output_dim = self.cfg.output_singer_dim

        self.nn = nn.Embedding(
            num_embeddings=cfg.singer_table_size,
            embedding_dim=self.output_dim,
            padding_idx=None,
        )

    def forward(self, x):
        # x: (N, 1) -> (N, 1, output_dim)
        return self.nn(x)


class ConditionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.merge_mode = cfg.merge_mode

        if cfg.use_whisper:
            self.whisper_encoder = ContentEncoder(
                self.cfg.whisper_dim, self.cfg.content_encoder_dim
            )

        if cfg.use_contentvec:
            self.contentvec_encoder = ContentEncoder(
                self.cfg.contentvec_dim, self.cfg.content_encoder_dim
            )

        if cfg.use_mert:
            self.mert_encoder = ContentEncoder(
                self.cfg.mert_dim, self.cfg.content_encoder_dim
            )

        if cfg.use_wenet:
            self.wenet_encoder = ContentEncoder(
                self.cfg.wenet_dim, self.cfg.content_encoder_dim
            )

        self.melody_encoder = MelodyEncoder(self.cfg)
        self.loudness_encoder = LoudnessEncoder(self.cfg)
        if cfg.use_spkid:
            self.singer_encoder = SingerEncoder(self.cfg)

    def forward(self, x):
        outputs = []

        if "frame_pitch" in x.keys():
            if "frame_uv" not in x.keys():
                x["frame_uv"] = None
            pitch_enc_out = self.melody_encoder(
                x["frame_pitch"], uv=x["frame_uv"], length=x["target_len"]
            )
            outputs.append(pitch_enc_out)

        if "frame_energy" in x.keys():
            loudness_enc_out = self.loudness_encoder(x["frame_energy"])
            outputs.append(loudness_enc_out)

        if "whisper_feat" in x.keys():
            # whisper_feat: [b, T, 1024]
            whiser_enc_out = self.whisper_encoder(
                x["whisper_feat"], length=x["target_len"]
            )
            outputs.append(whiser_enc_out)
            seq_len = whiser_enc_out.shape[1]

        if "contentvec_feat" in x.keys():
            contentvec_enc_out = self.contentvec_encoder(
                x["contentvec_feat"], length=x["target_len"]
            )
            outputs.append(contentvec_enc_out)
            seq_len = contentvec_enc_out.shape[1]

        if "mert_feat" in x.keys():
            mert_enc_out = self.mert_encoder(x["mert_feat"], length=x["target_len"])
            outputs.append(mert_enc_out)
            seq_len = mert_enc_out.shape[1]

        if "wenet_feat" in x.keys():
            wenet_enc_out = self.wenet_encoder(x["wenet_feat"], length=x["target_len"])
            outputs.append(wenet_enc_out)
            seq_len = wenet_enc_out.shape[1]

        if "spk_id" in x.keys():
            speaker_enc_out = self.singer_encoder(x["spk_id"])  # [b, 1, 384]
            assert (
                "whisper_feat" in x.keys()
                or "contentvec_feat" in x.keys()
                or "mert_feat" in x.keys()
                or "wenet_feat" in x.keys()
            )
            singer_info = speaker_enc_out.expand(-1, seq_len, -1)
            outputs.append(singer_info)

        encoder_output = None
        if self.merge_mode == "concat":
            encoder_output = torch.cat(outputs, dim=-1)
        if self.merge_mode == "add":
            # (#modules, N, seq_len, output_dim)
            outputs = torch.cat([out[None, :, :, :] for out in outputs], dim=0)
            # (N, seq_len, output_dim)
            encoder_output = torch.sum(outputs, dim=0)

        return encoder_output
