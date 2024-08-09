# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 16:57:31 2021.

@author: Keqi Deng (UCAS)
"""

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
import torch
from torch import nn


class ContextualBlockEncoderLayer(nn.Module):
    """Contexutal Block Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        total_layer_num (int): Total number of layers
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        total_layer_num,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an EncoderLayer object."""
        super(ContextualBlockEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.total_layer_num = total_layer_num
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(
        self,
        x,
        mask,
        infer_mode=False,
        past_ctx=None,
        next_ctx=None,
        is_short_segment=False,
        layer_idx=0,
        cache=None,
    ):
        """Calculate forward propagation."""
        if self.training or not infer_mode:
            return self.forward_train(x, mask, past_ctx, next_ctx, layer_idx, cache)
        else:
            return self.forward_infer(
                x, mask, past_ctx, next_ctx, is_short_segment, layer_idx, cache
            )

    def forward_train(
        self, x, mask, past_ctx=None, next_ctx=None, layer_idx=0, cache=None
    ):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            past_ctx (torch.Tensor): Previous contexutal vector
            next_ctx (torch.Tensor): Next contexutal vector
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
            cur_ctx (torch.Tensor): Current contexutal vector
            next_ctx (torch.Tensor): Next contexutal vector
            layer_idx (int): layer index number

        """
        nbatch = x.size(0)
        nblock = x.size(1)

        if past_ctx is not None:
            if next_ctx is None:
                # store all context vectors in one tensor
                next_ctx = past_ctx.new_zeros(
                    nbatch, nblock, self.total_layer_num, x.size(-1)
                )
            else:
                x[:, :, 0] = past_ctx[:, :, layer_idx]

        # reshape ( nbatch, nblock, block_size + 2, dim )
        #     -> ( nbatch * nblock, block_size + 2, dim )
        x = x.view(-1, x.size(-2), x.size(-1))
        if mask is not None:
            mask = mask.view(-1, mask.size(-2), mask.size(-1))

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        layer_idx += 1
        # reshape ( nbatch * nblock, block_size + 2, dim )
        #       -> ( nbatch, nblock, block_size + 2, dim )
        x = x.view(nbatch, -1, x.size(-2), x.size(-1)).squeeze(1)
        if mask is not None:
            mask = mask.view(nbatch, -1, mask.size(-2), mask.size(-1)).squeeze(1)

        if next_ctx is not None and layer_idx < self.total_layer_num:
            next_ctx[:, 0, layer_idx, :] = x[:, 0, -1, :]
            next_ctx[:, 1:, layer_idx, :] = x[:, 0:-1, -1, :]

        return x, mask, False, next_ctx, next_ctx, layer_idx

    def forward_infer(
        self,
        x,
        mask,
        past_ctx=None,
        next_ctx=None,
        is_short_segment=False,
        layer_idx=0,
        cache=None,
    ):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            past_ctx (torch.Tensor): Previous contexutal vector
            next_ctx (torch.Tensor): Next contexutal vector
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
            cur_ctx (torch.Tensor): Current contexutal vector
            next_ctx (torch.Tensor): Next contexutal vector
            layer_idx (int): layer index number

        """
        nbatch = x.size(0)
        nblock = x.size(1)
        # if layer_idx == 0, next_ctx has to be None
        if layer_idx == 0:
            assert next_ctx is None
            next_ctx = x.new_zeros(nbatch, self.total_layer_num, x.size(-1))

        # reshape ( nbatch, nblock, block_size + 2, dim )
        #     -> ( nbatch * nblock, block_size + 2, dim )
        x = x.view(-1, x.size(-2), x.size(-1))
        if mask is not None:
            mask = mask.view(-1, mask.size(-2), mask.size(-1))

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        # reshape ( nbatch * nblock, block_size + 2, dim )
        #       -> ( nbatch, nblock, block_size + 2, dim )
        x = x.view(nbatch, nblock, x.size(-2), x.size(-1))
        if mask is not None:
            mask = mask.view(nbatch, nblock, mask.size(-2), mask.size(-1))

        # Propagete context information (the last frame of each block)
        # to the first frame
        # of the next block

        if not is_short_segment:
            if past_ctx is None:
                # First block of an utterance
                x[:, 0, 0, :] = x[:, 0, -1, :]
            else:
                x[:, 0, 0, :] = past_ctx[:, layer_idx, :]
            if nblock > 1:
                x[:, 1:, 0, :] = x[:, 0:-1, -1, :]
            next_ctx[:, layer_idx, :] = x[:, -1, -1, :]
        else:
            next_ctx = None

        return x, mask, True, past_ctx, next_ctx, is_short_segment, layer_idx + 1
