# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(output)


# import torch
# import torch.nn as nn
# import math

# class SinePositionalEmbedding(nn.Module):
#     def __init__(
#         self,
#         dim_model: int,
#         dropout: float = 0.0,
#         scale: bool = False,
#         alpha: bool = False,
#     ):
#         super().__init__()
#         self.dim_model = dim_model
#         self.x_scale = math.sqrt(dim_model) if scale else 1.0
#         self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
#         self.dropout = torch.nn.Dropout(p=dropout)

#         self.reverse = False
#         self.pe = None
#         self.extend_pe(torch.zeros(1, 4000))

#     def extend_pe(self, x):
#         """Reset the positional encodings."""
#         if self._pe_needs_extension(x):
#             self.pe = self._generate_positional_encodings(x)

#     def _pe_needs_extension(self, x):
#         return self.pe is None or self.pe.size(1) < x.size(1) or self.pe.dtype != x.dtype or self.pe.device != x.device

#     def _generate_positional_encodings(self, x):
#         pe = torch.zeros(x.size(1), self.dim_model)
#         position = self._get_position_tensor(x)
#         div_term = self._get_div_term()
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0).to(device=x.device, dtype=x.dtype).detach()

#     def _get_position_tensor(self, x):
#         position = torch.arange(x.size(1), dtype=torch.float32).unsqueeze(1)
#         return position.flip(0) if self.reverse else position

#     def _get_div_term(self):
#         return torch.exp(torch.arange(0, self.dim_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.dim_model))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         self.extend_pe(x)
#         output = x.unsqueeze(-1) if x.ndim == 2 else x
#         output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
#         return self.dropout(output)
