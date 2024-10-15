# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch

from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.nets_utils import rename_state_dict
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.vgg2l import VGG2L
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.attention import MultiHeadedAttention
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.dynamic_conv import DynamicConvolution
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.dynamic_conv2d import DynamicConvolution2D
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.embedding import PositionalEncoding
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.encoder_layer import EncoderLayer
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.layer_norm import LayerNorm
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.lightconv import LightweightConvolution
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.lightconv2d import LightweightConvolution2D
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.multi_layer_conv import Conv1dLinear, MultiLayeredConv1d
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.repeat import repeat
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.subsampling import Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        conv_wshare (int): The number of kernel of convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]): Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in self_attention_layer_type
            == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): Whether to use bias in convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        selfattention_layer_type (str): Encoder attention layer type.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length="11",
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        selfattention_layer_type="selfattn",
        padding_idx=-1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d-scaled-pos-enc":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 6
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 8
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )
        if selfattention_layer_type in [
            "selfattn",
            "rel_selfattn",
            "legacy_rel_selfattn",
        ]:
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = [
                (
                    attention_heads,
                    attention_dim,
                    attention_dropout_rate,
                )
            ] * num_blocks
        elif selfattention_layer_type == "lightconv":
            logging.info("encoder self-attention layer type = lightweight convolution")
            encoder_selfattn_layer = LightweightConvolution
            encoder_selfattn_layer_args = [
                (
                    conv_wshare,
                    attention_dim,
                    attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),
                    False,
                    conv_usebias,
                )
                for lnum in range(num_blocks)
            ]
        elif selfattention_layer_type == "lightconv2d":
            logging.info(
                "encoder self-attention layer "
                "type = lightweight convolution 2-dimentional"
            )
            encoder_selfattn_layer = LightweightConvolution2D
            encoder_selfattn_layer_args = [
                (
                    conv_wshare,
                    attention_dim,
                    attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),
                    False,
                    conv_usebias,
                )
                for lnum in range(num_blocks)
            ]
        elif selfattention_layer_type == "dynamicconv":
            logging.info("encoder self-attention layer type = dynamic convolution")
            encoder_selfattn_layer = DynamicConvolution
            encoder_selfattn_layer_args = [
                (
                    conv_wshare,
                    attention_dim,
                    attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),
                    False,
                    conv_usebias,
                )
                for lnum in range(num_blocks)
            ]
        elif selfattention_layer_type == "dynamicconv2d":
            logging.info(
                "encoder self-attention layer type = dynamic convolution 2-dimentional"
            )
            encoder_selfattn_layer = DynamicConvolution2D
            encoder_selfattn_layer_args = [
                (
                    conv_wshare,
                    attention_dim,
                    attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),
                    False,
                    conv_usebias,
                )
                for lnum in range(num_blocks)
            ]
        else:
            raise NotImplementedError(selfattention_layer_type)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args[lnum]),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def get_positionwise_layer(
        self,
        positionwise_layer_type="linear",
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        positionwise_conv_kernel_size=1,
    ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.

        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
