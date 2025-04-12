# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ALL_COMPLETED
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F

from models.codec.amphion_codec.quantize import ResidualVQ
from models.codec.amphion_codec.vocos import VocosBackbone


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def compute_codebook_perplexity(indices, codebook_size):
    indices = indices.flatten()
    prob = torch.bincount(indices, minlength=codebook_size).float() / indices.size(0)
    perp = torch.exp(-torch.sum(prob * torch.log(prob + 1e-10)))
    return perp


class CocoContentStyle(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        num_quantizers=1,
        quantizer_type="fvq",
        use_whisper=True,
        use_chromagram=True,
        construct_only_for_quantizer=False,
        cfg=None,
    ):
        super().__init__()

        assert cfg is not None
        self.cfg = cfg

        codebook_size = getattr(cfg, "codebook_size", codebook_size)
        hidden_size = getattr(cfg, "hidden_size", hidden_size)
        codebook_dim = getattr(cfg, "codebook_dim", codebook_dim)
        num_quantizers = getattr(cfg, "num_quantizers", num_quantizers)
        quantizer_type = getattr(cfg, "quantizer_type", quantizer_type)

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.num_quantizers = num_quantizers
        self.quantizer_type = quantizer_type

        if use_whisper:
            self.whisper_input_layer = nn.Linear(self.cfg.whisper_dim, hidden_size)
        if use_chromagram:
            self.chromagram_input_layer = nn.Linear(
                self.cfg.chromagram_dim, hidden_size
            )

        downsample_rate = getattr(cfg, "downsample_rate", 1)
        if downsample_rate > 1:
            self.do_downsample = True
            assert np.log2(downsample_rate).is_integer()

            down_layers = []
            up_layers = []
            for _ in range(int(np.log2(downsample_rate))):
                down_layers.extend(
                    [
                        nn.Conv1d(
                            hidden_size,
                            hidden_size,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GELU(),
                    ]
                )
                up_layers.extend(
                    [
                        nn.ConvTranspose1d(
                            hidden_size, hidden_size, kernel_size=4, stride=2, padding=1
                        ),
                        nn.GELU(),
                    ]
                )
            self.downsample_layers = nn.Sequential(*down_layers)
            self.upsample_layers = nn.Sequential(*up_layers)

        else:
            self.do_downsample = False

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.cfg.encoder.vocos_dim,
                intermediate_dim=self.cfg.encoder.vocos_intermediate_dim,
                num_layers=self.cfg.encoder.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.cfg.encoder.vocos_dim, self.hidden_size),
        )

        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_type=quantizer_type,
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )

        if not construct_only_for_quantizer:
            self.decoder = nn.Sequential(
                VocosBackbone(
                    input_channels=self.hidden_size,
                    dim=self.cfg.decoder.vocos_dim,
                    intermediate_dim=self.cfg.decoder.vocos_intermediate_dim,
                    num_layers=self.cfg.decoder.vocos_num_layers,
                    adanorm_num_embeddings=None,
                ),
                nn.Linear(self.cfg.decoder.vocos_dim, self.hidden_size),
            )

            if use_whisper:
                self.whisper_output_layer = nn.Linear(
                    self.hidden_size, self.cfg.whisper_dim
                )
            if use_chromagram:
                self.chromagram_output_layer = nn.Linear(
                    self.hidden_size, self.cfg.chromagram_dim
                )

        self.reset_parameters()

    def forward(
        self,
        whisper_feats,
        chromagram_feats,
        return_for_quantizer=False,
    ):
        """
        Args:
            whisper_feats: [B, T, 1024]
            chromagram_feats: [B, T, 24]
        Returns:
            whisper_rec: [B, T, 1024]
            chromagram_rec: [B, T, 24]
            codebook_loss: float
            all_indices: [N, B, T] or [B, T] if num_of_quantizers == 1
        """
        T = whisper_feats.shape[1]

        # [B, T, D]
        x = self.whisper_input_layer(whisper_feats) + self.chromagram_input_layer(
            chromagram_feats
        )
        # print("Before downsample:", x.shape)

        # ====== Downsample ======
        if self.do_downsample:
            x = self.downsample_layers(x.transpose(1, 2)).transpose(1, 2)

        # print("After downsample:", x.shape)

        # ====== Encoder ======
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D] -> [B, D, T]

        # ====== Quantizer ======
        (
            quantized_out,  # [B, D, T]
            all_indices,  # [num_of_quantizers, B, T]
            all_commit_losses,  # [num_of_quantizers]
            all_codebook_losses,  # [num_of_quantizers]
            _,
        ) = self.quantizer(x)

        if return_for_quantizer:
            if all_indices.shape[0] == 1:
                return all_indices.squeeze(0), quantized_out.transpose(1, 2)
            return all_indices, quantized_out.transpose(1, 2)

        # ====== Decoder ======
        x_rec = self.decoder(quantized_out)  # [B, T, D]

        # ====== Upsample ======
        if self.do_downsample:
            x_rec = self.upsample_layers(x_rec.transpose(1, 2)).transpose(1, 2)

        # print("After upsample:", x_rec.shape)

        # Ensure output dimensions match input
        if x_rec.shape[1] >= T:  # Check time dimension
            x_rec = x_rec[:, :T, :]
        else:
            padding_frames = T - x_rec.shape[1]
            last_frame = x_rec[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            x_rec = torch.cat([x_rec, padding], dim=1)

        # ====== Loss ======
        whisper_rec = self.whisper_output_layer(x_rec)  # [B, T, 1024]
        chromagram_rec = self.chromagram_output_layer(x_rec)  # [B, T, 24]

        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        all_indices = all_indices

        return whisper_rec, chromagram_rec, codebook_loss, all_indices

    def quantize(self, whisper_feats, chromagram_feats):
        """
        Args:
            whisper_feats: [B, T, 1024]
            chromagram_feats: [B, T, 24]
        Returns:
            all_indices: [N, B, T], or [B, T] if num_of_quantizers == 1
            quantized_out: [B, D, T]
        """
        all_indices, quantized_out = self.forward(
            whisper_feats,
            chromagram_feats,
            return_for_quantizer=True,
        )
        return all_indices, quantized_out

    def reset_parameters(self):
        self.apply(init_weights)


class CocoContent(CocoContentStyle):
    def __init__(
        self,
        cfg,
        use_whisper=True,
        use_chromagram=False,
        construct_only_for_quantizer=False,
    ):
        super().__init__(
            cfg=cfg,
            use_whisper=use_whisper,
            use_chromagram=use_chromagram,
            construct_only_for_quantizer=construct_only_for_quantizer,
        )

    def forward(
        self,
        whisper_feats,
        return_for_quantizer=False,
    ):
        """
        Args:
            whisper_feats: [B, T, 1024]
        Returns:
            whisper_rec: [B, T, 1024]
            codebook_loss: float
            all_indices: [N, B, T]
        """
        T = whisper_feats.shape[1]

        # [B, T, D]
        x = self.whisper_input_layer(whisper_feats)

        # ====== Downsample ======
        if self.do_downsample:
            x = self.downsample_layers(x.transpose(1, 2)).transpose(1, 2)

        # ====== Encoder ======
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D] -> [B, D, T]

        # ====== Quantizer ======
        (
            quantized_out,  # [B, D, T]
            all_indices,  # [num_of_quantizers, B, T]
            all_commit_losses,  # [num_of_quantizers]
            all_codebook_losses,  # [num_of_quantizers]
            _,
        ) = self.quantizer(x)

        if return_for_quantizer:
            if all_indices.shape[0] == 1:
                return all_indices.squeeze(0), quantized_out.transpose(1, 2)
            return all_indices, quantized_out.transpose(1, 2)

        # ====== Decoder ======
        x_rec = self.decoder(quantized_out)  # [B, T, D]

        # ====== Upsample ======
        if self.do_downsample:
            x_rec = self.upsample_layers(x_rec.transpose(1, 2)).transpose(1, 2)

        # Ensure output dimensions match input
        if x_rec.shape[1] >= T:  # Check time dimension
            x_rec = x_rec[:, :T, :]
        else:
            padding_frames = T - x_rec.shape[1]
            last_frame = x_rec[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            x_rec = torch.cat([x_rec, padding], dim=1)

        # ====== Loss ======
        whisper_rec = self.whisper_output_layer(x_rec)  # [B, T, 1024]

        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        all_indices = all_indices

        return whisper_rec, codebook_loss, all_indices

    def quantize(self, whisper_feats):
        all_indices, quantized_out = self.forward(
            whisper_feats, return_for_quantizer=True
        )
        return all_indices, quantized_out


class CocoStyle(CocoContentStyle):
    def __init__(
        self,
        cfg,
        use_whisper=False,
        use_chromagram=True,
        construct_only_for_quantizer=False,
    ):
        super().__init__(
            cfg=cfg,
            use_whisper=use_whisper,
            use_chromagram=use_chromagram,
            construct_only_for_quantizer=construct_only_for_quantizer,
        )

    def forward(
        self,
        chromagram_feats,
        return_for_quantizer=False,
    ):
        """
        Args:
            chromagram_feats: [B, T, 24]
        Returns:
            chromagram_rec: [B, T, 24]
            codebook_loss: float
            all_indices: [N, B, T]
        """
        T = chromagram_feats.shape[1]

        # [B, T, D]
        x = self.chromagram_input_layer(chromagram_feats)

        # ====== Downsample ======
        if self.do_downsample:
            x = self.downsample_layers(x.transpose(1, 2)).transpose(1, 2)

        # ====== Encoder ======
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D] -> [B, D, T]

        # ====== Quantizer ======
        (
            quantized_out,  # [B, D, T]
            all_indices,  # [num_of_quantizers, B, T]
            all_commit_losses,  # [num_of_quantizers]
            all_codebook_losses,  # [num_of_quantizers]
            _,
        ) = self.quantizer(x)

        if return_for_quantizer:
            if all_indices.shape[0] == 1:
                return all_indices.squeeze(0), quantized_out.transpose(1, 2)
            return all_indices, quantized_out.transpose(1, 2)

        # ====== Decoder ======
        x_rec = self.decoder(quantized_out)  # [B, T, D]

        # ====== Upsample ======
        if self.do_downsample:
            x_rec = self.upsample_layers(x_rec.transpose(1, 2)).transpose(1, 2)

        # Ensure output dimensions match input
        if x_rec.shape[1] >= T:  # Check time dimension
            x_rec = x_rec[:, :T, :]
        else:
            padding_frames = T - x_rec.shape[1]
            last_frame = x_rec[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            x_rec = torch.cat([x_rec, padding], dim=1)

        # ====== Loss ======
        chromagram_rec = self.chromagram_output_layer(x_rec)  # [B, T, 24]

        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        all_indices = all_indices

        return chromagram_rec, codebook_loss, all_indices

    def quantize(self, chromagram_feats):
        all_indices, quantized_out = self.forward(
            chromagram_feats, return_for_quantizer=True
        )
        return all_indices, quantized_out


# if __name__ == "__main__":
#     from utils.util import JsonHParams

#     cfg = JsonHParams(
#         **{
#             "whisper_dim": 1024,
#             "chromagram_dim": 24,
#             "global_speaker_encoder": {
#                 "input_dim": 128,  # Eg: n_mels
#                 "hidden_size": 512,  # 768 for emilia298k
#                 "num_hidden_layers": 4,  # 6 for emilia298k
#                 "num_attention_heads": 8,
#             },
#         }
#     )
#     model = Coco(cfg=cfg)

#     x = torch.randn(2, 150, 1024)
#     tone_height = torch.randn(2)
#     mels = torch.randn(2, 150, 128)
#     mel_mask = torch.ones(2, 150)

#     x_rec, codebook_loss, all_indices, auxillary_pred_outputs = model(
#         x, tone_height, mels, mel_mask
#     )
#     print(x_rec.shape, codebook_loss, all_indices.shape)
#     for k, v in auxillary_pred_outputs.items():
#         print(k, v.shape)
