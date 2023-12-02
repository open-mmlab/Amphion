# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def normalization(channels: int, groups: int = 32):
    r"""Make a standard normalization layer, i.e. GroupNorm.

    Args:
        channels: number of input channels.
        groups: number of groups for group normalization.

    Returns:
        a ``nn.Module`` for normalization.
    """
    assert groups > 0, f"invalid number of groups: {groups}"
    return nn.GroupNorm(groups, channels)


def Linear(*args, **kwargs):
    r"""Wrapper of ``nn.Linear`` with kaiming_normal_ initialization."""
    layer = nn.Linear(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Conv1d(*args, **kwargs):
    r"""Wrapper of ``nn.Conv1d`` with kaiming_normal_ initialization."""
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Conv2d(*args, **kwargs):
    r"""Wrapper of ``nn.Conv2d`` with kaiming_normal_ initialization."""
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def ConvNd(dims: int = 1, *args, **kwargs):
    r"""Wrapper of N-dimension convolution with kaiming_normal_ initialization.

    Args:
        dims: number of dimensions of the convolution.
    """
    if dims == 1:
        return Conv1d(*args, **kwargs)
    elif dims == 2:
        return Conv2d(*args, **kwargs)
    else:
        raise ValueError(f"invalid number of dimensions: {dims}")


def zero_module(module: nn.Module):
    r"""Zero out the parameters of a module and return it."""
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


def scale_module(module: nn.Module, scale):
    r"""Scale the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor: torch.Tensor):
    r"""Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=tuple(range(1, tensor.dim())))


def append_dims(x, target_dims):
    r"""Appends dimensions to the end of a tensor until
    it has target_dims dimensions.
    """
    dims_to_append = target_dims - x.dim()
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.dim()} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x, count=1):
    r"""Appends ``count`` zeros to the end of a tensor along the last dimension."""
    assert count > 0, f"invalid count: {count}"
    return torch.cat([x, x.new_zeros((*x.size()[:-1], count))], dim=-1)


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)
