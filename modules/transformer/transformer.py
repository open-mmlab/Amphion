# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import partial
from typing import Any, Callable, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modules.norms import AdaptiveLayerNorm, LayerNorm, BalancedBasicNorm, IdentityNorm
from modules.transformer import MultiheadAttention
from modules.general.scaling import BalancedDoubleSwish


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        linear1_self_attention_cls: nn.Module = nn.Linear,
        linear2_self_attention_cls: nn.Module = nn.Linear,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        layer_norm_cls: nn.Module = LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )

        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model, **factory_kwargs
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)

        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        if layer_norm_cls == IdentityNorm:
            norm2 = BalancedBasicNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding),
                src_mask,
                src_key_padding_mask,
            )
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask),
                stage_embedding,
            )
            x = self.norm2(x + self._ff_block(x), stage_embedding)

        if is_src_tuple:
            return (x, stage_embedding)
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_layer_states: bool = False,
    ) -> Tensor:
        # Pass the input through the encoder layers
        output = src
        layer_states = [] if return_layer_states else None

        for mod in self.layers:
            output = self._apply_module(
                mod, output, mask, src_key_padding_mask, layer_states
            )

        if self.norm is not None:
            output = self.norm(output)

        return (layer_states, output) if return_layer_states else output

    def _apply_module(self, module, output, mask, key_padding_mask, layer_states):
        # Apply a single transformer module
        output = module(output, src_mask=mask, src_key_padding_mask=key_padding_mask)
        if layer_states is not None:
            layer_states.append(output)
        return output


class TransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        linear1_self_attention_cls: nn.Module = nn.Linear,
        linear2_self_attention_cls: nn.Module = nn.Linear,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        layer_norm_cls: nn.Module = LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model, **factory_kwargs
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm1, self.norm2, self.norm3 = self._init_norm_layers(
            d_model, layer_norm_cls, layer_norm_eps, adaptive_layer_norm, factory_kwargs
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt_is_tuple = False
        if isinstance(tgt, tuple):
            x, stage_embedding = tgt
            tgt_is_tuple = True
        else:
            x, stage_embedding = tgt, None

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding), tgt_mask, tgt_key_padding_mask
            )
            x = x + self._mha_block(
                self.norm2(x, stage_embedding),
                memory,
                memory_mask,
                memory_key_padding_mask,
            )
            x = x + self._ff_block(self.norm3(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask),
                stage_embedding,
            )
            x = self.norm2(
                x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                stage_embedding,
            )
            x = self.norm3(x + self._ff_block(x), stage_embedding)

        if tgt_is_tuple:
            return (x, stage_embedding)
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def _get_activation_fn(self, activation):
        if isinstance(activation, str):
            return _get_activation_fn(activation)
        elif callable(activation):
            return activation
        else:
            raise ValueError("Unsupported activation type")

    def _init_norm_layers(
        self,
        d_model,
        layer_norm_cls,
        layer_norm_eps,
        adaptive_layer_norm,
        factory_kwargs,
    ):
        if adaptive_layer_norm:
            return (
                AdaptiveLayerNorm(
                    d_model,
                    layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs),
                ),
                AdaptiveLayerNorm(
                    d_model,
                    layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs),
                ),
                AdaptiveLayerNorm(
                    d_model,
                    layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs),
                ),
            )
        else:
            return (
                layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs),
                layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs),
                (
                    layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
                    if layer_norm_cls != IdentityNorm
                    else BalancedBasicNorm(
                        d_model, eps=layer_norm_eps, **factory_kwargs
                    )
                ),
            )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)
