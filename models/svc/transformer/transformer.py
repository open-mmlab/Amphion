# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dropout = self.cfg.dropout
        nhead = self.cfg.n_heads
        nlayers = self.cfg.n_layers
        input_dim = self.cfg.input_dim
        output_dim = self.cfg.output_dim

        d_model = input_dim
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.output_mlp = nn.Linear(d_model, output_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, seq_len, input_dim)
        Returns:
            output: (N, seq_len, output_dim)
        """
        # (N, seq_len, d_model)
        src = self.pos_encoder(x)
        # model_stats["pos_embedding"] = x
        # (N, seq_len, d_model)
        output = self.transformer_encoder(src)
        # (N, seq_len, output_dim)
        output = self.output_mlp(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Assume that x is (seq_len, N, d)
        # pe = torch.zeros(max_len, 1, d_model)
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Assume that x in (N, seq_len, d)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [N, seq_len, d]
        """
        # Old: Assume that x is (seq_len, N, d), and self.pe is (max_len, 1, d_model)
        # x = x + self.pe[: x.size(0)]

        # Now: self.pe is (1, max_len, d)
        x = x + self.pe[:, : x.size(1), :]

        return self.dropout(x)
