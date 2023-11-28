# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from modules.diffusion import BiDilConv
from modules.encoder.position_encoder import PositionEncoder


class DiffusionWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.diff_cfg = cfg.model.diffusion

        self.diff_encoder = PositionEncoder(
            d_raw_emb=self.diff_cfg.step_encoder.dim_raw_embedding,
            d_out=self.diff_cfg.bidilconv.base_channel,
            d_mlp=self.diff_cfg.step_encoder.dim_hidden_layer,
            activation_function=self.diff_cfg.step_encoder.activation,
            n_layer=self.diff_cfg.step_encoder.num_layer,
            max_period=self.diff_cfg.step_encoder.max_period,
        )

        # FIXME: Only support BiDilConv now for debug
        if self.diff_cfg.model_type.lower() == "bidilconv":
            self.neural_network = BiDilConv(
                input_channel=self.cfg.preprocess.n_mel, **self.diff_cfg.bidilconv
            )
        else:
            raise ValueError(
                f"Unsupported diffusion model type: {self.diff_cfg.model_type}"
            )

    def forward(self, x, t, c):
        """
        Args:
            x: [N, T, mel_band] of mel spectrogram
            t: Diffusion time step with shape of [N]
            c: [N, T, conditioner_size] of conditioner

        Returns:
            [N, T, mel_band] of mel spectrogram
        """

        assert (
            x.size()[:-1] == c.size()[:-1]
        ), "x mismatch with c, got \n x: {} \n c: {}".format(x.size(), c.size())
        assert x.size(0) == t.size(
            0
        ), "x mismatch with t, got \n x: {} \n t: {}".format(x.size(), t.size())
        assert t.dim() == 1, "t must be 1D tensor, got {}".format(t.dim())

        N, T, mel_band = x.size()

        x = x.transpose(1, 2).contiguous()  # [N, mel_band, T]
        c = c.transpose(1, 2).contiguous()  # [N, conditioner_size, T]
        t = self.diff_encoder(t).contiguous()  # [N, base_channel]

        h = self.neural_network(x, t, c)
        h = h.transpose(1, 2).contiguous()  # [N, T, mel_band]

        assert h.size() == (
            N,
            T,
            mel_band,
        ), "h mismatch with input x, got \n h: {} \n x: {}".format(
            h.size(), (N, T, mel_band)
        )
        return h
