# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from https://github.com/lifeiteng/vall-e

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, dim_model: int, vocab_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(vocab_size, dim_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def forward(self, x: torch.Tensor):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        return x
