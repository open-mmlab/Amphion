# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/models.py
import copy
import torch
from torch import nn
from torch.nn import functional as F

from utils.util import *
from utils.f0 import f0_to_coarse

from modules.transformer.attentions import Encoder
from models.tts.vits.vits import ResidualCouplingBlock, PosteriorEncoder
from models.vocoders.gan.generator.bigvgan import BigVGAN
from models.vocoders.gan.generator.hifigan import HiFiGAN
from models.vocoders.gan.generator.nsfhifigan import NSFHiFiGAN
from models.vocoders.gan.generator.melgan import MelGAN
from models.vocoders.gan.generator.apnet import APNet
from modules.encoder.condition_encoder import ConditionEncoder


def slice_pitch_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
    return ret, ret_pitch, ids_str


class ContentEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        kernel_size,
        n_layers,
        gin_channels=0,
        filter_channels=None,
        n_heads=None,
        p_dropout=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # condition_encoder ver.
    def forward(self, x, x_mask, noice_scale=1):
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask

        return z, m, logs, x_mask


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self, spec_channels, segment_size, cfg):
        super().__init__()
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.cfg = cfg
        self.inter_channels = cfg.model.vits.inter_channels
        self.hidden_channels = cfg.model.vits.hidden_channels
        self.filter_channels = cfg.model.vits.filter_channels
        self.n_heads = cfg.model.vits.n_heads
        self.n_layers = cfg.model.vits.n_layers
        self.kernel_size = cfg.model.vits.kernel_size
        self.p_dropout = cfg.model.vits.p_dropout
        self.n_flow_layer = cfg.model.vits.n_flow_layer
        self.gin_channels = cfg.model.vits.gin_channels
        self.n_speakers = cfg.model.vits.n_speakers

        # f0
        self.n_bins = cfg.preprocess.pitch_bin
        self.f0_min = cfg.preprocess.f0_min
        self.f0_max = cfg.preprocess.f0_max

        # TODO: sort out the config
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)

        self.emb_g = nn.Embedding(self.n_speakers, self.gin_channels)

        self.enc_p = ContentEncoder(
            self.inter_channels,
            self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
        )

        assert cfg.model.generator in [
            "bigvgan",
            "hifigan",
            "melgan",
            "nsfhifigan",
            "apnet",
        ]
        self.dec_name = cfg.model.generator
        temp_cfg = copy.deepcopy(cfg)
        temp_cfg.preprocess.n_mel = self.inter_channels
        if cfg.model.generator == "bigvgan":
            temp_cfg.model.bigvgan = cfg.model.generator_config.bigvgan
            self.dec = BigVGAN(temp_cfg)
        elif cfg.model.generator == "hifigan":
            temp_cfg.model.hifigan = cfg.model.generator_config.hifigan
            self.dec = HiFiGAN(temp_cfg)
        elif cfg.model.generator == "melgan":
            temp_cfg.model.melgan = cfg.model.generator_config.melgan
            self.dec = MelGAN(temp_cfg)
        elif cfg.model.generator == "nsfhifigan":
            temp_cfg.model.nsfhifigan = cfg.model.generator_config.nsfhifigan
            self.dec = NSFHiFiGAN(temp_cfg)  # TODO: nsf need f0
        elif cfg.model.generator == "apnet":
            temp_cfg.model.apnet = cfg.model.generator_config.apnet
            self.dec = APNet(temp_cfg)

        self.enc_q = PosteriorEncoder(
            self.spec_channels,
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.gin_channels,
        )

        self.flow = ResidualCouplingBlock(
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            self.n_flow_layer,
            gin_channels=self.gin_channels,
        )

    def forward(self, data):
        """VitsSVC forward function.

        Args:
            data (dict): condition data & audio data, including:
            B: batch size, T: target length
            {
                "spk_id": [B, singer_table_size]
                "target_len": [B]
                "mask": [B, T, 1]
                "mel": [B, T, n_mel]
                "linear": [B, T, n_fft // 2 + 1]
                "frame_pitch": [B, T]
                "frame_uv": [B, T]
                "audio": [B, audio_len]
                "audio_len": [B]
                "contentvec_feat": [B, T, contentvec_dim]
                "whisper_feat": [B, T, whisper_dim]
                ...
            }
        """

        # TODO: elegantly handle the dimensions
        spec = data["linear"].transpose(1, 2)

        g = data["spk_id"]
        g = self.emb_g(g).transpose(1, 2)

        c_lengths = data["target_len"]
        spec_lengths = data["target_len"]
        f0 = data["frame_pitch"]

        # condition_encoder ver.
        x = self.condition_encoder(data).transpose(1, 2)
        x_mask = torch.unsqueeze(sequence_mask(c_lengths, f0.size(1)), 1).to(x.dtype)

        # prior encoder
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask)
        # posterior encoder
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        # flow
        z_p = self.flow(z, spec_mask, g=g)
        z_slice, pitch_slice, ids_slice = rand_slice_segments_with_pitch(
            z, f0, spec_lengths, self.segment_size
        )

        if self.dec_name == "nsfhifigan":
            o = self.dec(z_slice, f0=f0.float())
        elif self.dec_name == "apnet":
            _, _, _, _, o = self.dec(z_slice)
        else:
            o = self.dec(z_slice)

        outputs = {
            "y_hat": o,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "z_mask": data["mask"].transpose(1, 2),
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
        }
        return outputs

    @torch.no_grad()
    def infer(self, data, noise_scale=0.35, seed=52468):
        # c, f0, uv, g
        f0 = data["frame_pitch"]
        g = data["spk_id"]

        if f0.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(f0.size(0)) * f0.size(-1)).to(f0.device)

        if g.dim() == 1:
            g = g.unsqueeze(0)
        g = self.emb_g(g).transpose(1, 2)

        # condition_encoder ver.
        x = self.condition_encoder(data).transpose(1, 2)
        x_mask = torch.unsqueeze(sequence_mask(c_lengths, f0.size(1)), 1).to(x.dtype)

        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, noice_scale=noise_scale)
        z = self.flow(z_p, c_mask, g=g, reverse=True)

        if self.dec_name == "nsfhifigan":
            o = self.dec(z * c_mask, f0=f0.float())
        elif self.dec_name == "apnet":
            _, _, _, _, o = self.dec(z * c_mask)
        else:
            o = self.dec(z * c_mask)
        return o, f0
