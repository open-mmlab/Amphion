# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from modules.encoder.position_encoder import PositionEncoder
from modules.general.utils import append_dims, ConvNd, normalization, zero_module
from .attention import AttentionBlock
from .resblock import Downsample, ResBlock, Upsample


class UNet(nn.Module):
    r"""The full UNet model with attention and timestep embedding.

    Args:
        dims: determines if the signal is 1D (temporal), 2D(spatial).
        in_channels: channels in the input Tensor.
        model_channels: base channel count for the model.
        out_channels: channels in the output Tensor.
        num_res_blocks: number of residual blocks per downsample.
        channel_mult: channel multiplier for each level of the UNet.
        num_attn_blocks: number of attention blocks at place.
        attention_resolutions: a collection of downsample rates at which attention will
            take place. May be a set, list, or tuple. For example, if this contains 4,
            then at 4x downsampling, attention will be used.
        num_heads: the number of attention heads in each attention layer.
        num_head_channels: if specified, ignore num_heads and instead use a fixed
            channel width per attention head.
        d_context: if specified, use for cross-attention channel project.
        p_dropout: the dropout probability.
        use_self_attention: Apply self attention before cross attention.
        num_classes: if specified (as an int), then this model will be class-conditional
            with ``num_classes`` classes.
        use_extra_film: if specified, use an extra FiLM-like conditioning mechanism.
        d_emb: if specified, use for FiLM-like conditioning.
        use_scale_shift_norm: use a FiLM-like conditioning mechanism.
        resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        dims: int = 1,
        in_channels: int = 100,
        model_channels: int = 128,
        out_channels: int = 100,
        h_dim: int = 128,
        num_res_blocks: int = 1,
        channel_mult: tuple = (1, 2, 4),
        num_attn_blocks: int = 1,
        attention_resolutions: tuple = (1, 2, 4),
        num_heads: int = 1,
        num_head_channels: int = -1,
        d_context: int = None,
        context_hdim: int = 128,
        p_dropout: float = 0.0,
        num_classes: int = -1,
        use_extra_film: str = None,
        d_emb: int = None,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_attn_blocks = num_attn_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.d_context = d_context
        self.p_dropout = p_dropout
        self.num_classes = num_classes
        self.use_extra_film = use_extra_film
        self.d_emb = d_emb
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown

        time_embed_dim = model_channels * 4
        self.pos_enc = PositionEncoder(model_channels, time_embed_dim)

        assert (
            num_classes == -1 or use_extra_film is None
        ), "You cannot set both num_classes and use_extra_film."

        if self.num_classes > 0:
            # TODO: if used for singer, norm should be 1, correct?
            self.label_emb = nn.Embedding(num_classes, time_embed_dim, max_norm=1.0)
        elif use_extra_film is not None:
            assert (
                d_emb is not None
            ), "d_emb must be specified if use_extra_film is not None"
            assert use_extra_film in [
                "add",
                "concat",
            ], f"use_extra_film only supported by add or concat. Your input is {use_extra_film}"
            self.use_extra_film = use_extra_film
            self.film_emb = ConvNd(dims, d_emb, time_embed_dim, 1)
            if use_extra_film == "concat":
                time_embed_dim *= 2

        # Input blocks
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [UNetSequential(ConvNd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        p_dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    for _ in range(num_attn_blocks):
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                encoder_channels=d_context,
                                dims=dims,
                                h_dim=h_dim // (level + 1),
                                encoder_hdim=context_hdim,
                                p_dropout=p_dropout,
                            )
                        )
                self.input_blocks.append(UNetSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    UNetSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            p_dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle blocks
        self.middle_block = UNetSequential(
            ResBlock(
                ch,
                time_embed_dim,
                p_dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=d_context,
                dims=dims,
                h_dim=h_dim // (level + 1),
                encoder_hdim=context_hdim,
                p_dropout=p_dropout,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                p_dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in tuple(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        p_dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    for _ in range(num_attn_blocks):
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                encoder_channels=d_context,
                                dims=dims,
                                h_dim=h_dim // (level + 1),
                                encoder_hdim=context_hdim,
                                p_dropout=p_dropout,
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            p_dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(UNetSequential(*layers))
                self._feature_size += ch

        # Final proj out
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(ConvNd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        r"""Apply the model to an input batch.

        Args:
            x: an [N x C x ...] Tensor of inputs.
            timesteps: a 1-D batch of timesteps, i.e. [N].
            context: conditioning Tensor with shape of [N x ``d_context`` x ...] plugged
            in via cross attention.
            y: an [N] Tensor of labels, if **class-conditional**.
            an [N x ``d_emb`` x ...] Tensor if **film-embed conditional**.

        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        assert (y is None) or (
            (y is not None)
            and ((self.num_classes > 0) or (self.use_extra_film is not None))
        ), f"y must be specified if num_classes or use_extra_film is not None. \nGot num_classes: {self.num_classes}\t\nuse_extra_film: {self.use_extra_film}\t\n"

        hs = []
        emb = self.pos_enc(timesteps)
        emb = append_dims(emb, x.dim())

        if self.num_classes > 0:
            assert y.size() == (x.size(0),)
            emb = emb + self.label_emb(y)
        elif self.use_extra_film is not None:
            assert y.size() == (x.size(0), self.d_emb, *x.size()[2:])
            y = self.film_emb(y)
            if self.use_extra_film == "add":
                emb = emb + y
            elif self.use_extra_film == "concat":
                emb = torch.cat([emb, y], dim=1)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)


class UNetSequential(nn.Sequential):
    r"""A sequential module that passes embeddings to the children that support it."""

    def forward(self, x, emb=None, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
