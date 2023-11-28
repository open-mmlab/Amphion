# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, pow, sin
from torch.nn import Parameter


class Snake(nn.Module):
    r"""Implementation of a sine-based periodic activation function.
    Alpha is initialized to 1 by default, higher values means higher frequency.
    It will be trained along with the rest of your model.

    Args:
        in_features: shape of the input
        alpha: trainable parameter

    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input

    References:
        This activation function is from this paper by Liu Ziyin, Tilman Hartwig,
        Masahito Ueda: https://arxiv.org/abs/2006.08195

    Examples:
        >>> a1 = Snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        r"""Forward pass of the function. Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (ax)
        """

        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    r"""A modified Snake function which uses separate parameters for the magnitude
    of the periodic components. Alpha is initialized to 1 by default,
    higher values means higher frequency. Beta is initialized to 1 by default,
    higher values means higher magnitude. Both will be trained along with the
    rest of your model.

    Args:
        in_features: shape of the input
        alpha: trainable parameter that controls frequency
        beta: trainable parameter that controls magnitude

    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input

    References:
        This activation function is a modified version based on this paper by Liu Ziyin,
        Tilman Hartwig, Masahito Ueda: https://arxiv.org/abs/2006.08195

    Examples:
        >>> a1 = SnakeBeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        r"""Forward pass of the function. Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """

        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
