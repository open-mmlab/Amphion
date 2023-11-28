# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn

from modules.general.utils import Linear


class PositionEncoder(nn.Module):
    r"""Encoder of positional embedding, generates PE and then
    feed into 2 full-connected layers with ``SiLU``.

    Args:
        d_raw_emb: The dimension of raw embedding vectors.
        d_out: The dimension of output embedding vectors, default to ``d_raw_emb``.
        d_mlp: The dimension of hidden layer in MLP, default to ``d_raw_emb`` * 4.
        activation_function: The activation function used in MLP, default to ``SiLU``.
        n_layer: The number of layers in MLP, default to 2.
        max_period: controls the minimum frequency of the embeddings.
    """

    def __init__(
        self,
        d_raw_emb: int = 128,
        d_out: int = None,
        d_mlp: int = None,
        activation_function: str = "SiLU",
        n_layer: int = 2,
        max_period: int = 10000,
    ):
        super().__init__()

        self.d_raw_emb = d_raw_emb
        self.d_out = d_raw_emb if d_out is None else d_out
        self.d_mlp = d_raw_emb * 4 if d_mlp is None else d_mlp
        self.n_layer = n_layer
        self.max_period = max_period

        if activation_function.lower() == "silu":
            self.activation_function = "SiLU"
        elif activation_function.lower() == "relu":
            self.activation_function = "ReLU"
        elif activation_function.lower() == "gelu":
            self.activation_function = "GELU"
        else:
            raise ValueError("activation_function must be one of SiLU, ReLU, GELU")
        self.activation_function = activation_function

        tmp = [Linear(self.d_raw_emb, self.d_mlp), getattr(nn, activation_function)()]
        for _ in range(self.n_layer - 1):
            tmp.append(Linear(self.d_mlp, self.d_mlp))
            tmp.append(getattr(nn, activation_function)())
        tmp.append(Linear(self.d_mlp, self.d_out))

        self.out = nn.Sequential(*tmp)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        r"""Create and return sinusoidal timestep embeddings directly.

        Args:
            steps: a 1D Tensor of N indices, one per batch element.
                These may be fractional.

        Returns:
            an [N x ``d_out``] Tensor of positional embeddings.
        """

        half = self.d_raw_emb // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            / half
            * torch.arange(half, dtype=torch.float32, device=steps.device)
        )
        args = steps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_raw_emb % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return self.out(embedding)
