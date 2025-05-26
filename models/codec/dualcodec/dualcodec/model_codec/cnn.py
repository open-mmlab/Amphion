# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float = 0.0,
        adanorm_num_embeddings: Optional[int] = None,
        is_causal=False,
    ):
        """
            Initializes a DepthwiseSeparableAttention module.

        Args:
            dim (int): The input dimension of the model.
            intermediate_dim (int): The intermediate dimension of the model.
            layer_scale_init_value (float): The initial value for LayerScale. If it's
                greater than 0, LayerScale will be added to the model. Default: 0.
            adanorm_num_embeddings (Optional[int], optional): The number of embeddings in
                the AdaLayerNorm. Defaults to None.
        """
        super().__init__()
        self.is_causal = is_causal
        if not is_causal:
            self.dwconv = nn.Conv1d(
                dim, dim, kernel_size=7, padding=3, groups=dim
            )  # depthwise conv
        else:
            self.dwconv = nn.Conv1d(
                dim, dim, kernel_size=7, padding=0, groups=dim
            )  # depthwise conv

        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor, optional):
            cond_embedding_id (Optional[torch.Tensor], optional):

        Returns:
            torch.Tensor:

        Raises:
            None
        """
        residual = x
        if self.is_causal:
            x = torch.nn.functional.pad(x, (6, 0))
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        """
            Initializes the instance of the class.

        Args:
            num_embeddings (int): The number of embeddings in the embedding table.
            embedding_dim (int): The dimension of each embedding.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.shift = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):
            cond_embedding_id (torch.Tensor)

        Returns:
            torch.Tensor
        """
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x
