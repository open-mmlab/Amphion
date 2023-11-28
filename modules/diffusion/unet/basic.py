# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from abc import abstractmethod


class UNetBlock(nn.Module):
    r"""Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x, emb):
        r"""Apply the module to `x` given `emb` timestep embeddings."""
