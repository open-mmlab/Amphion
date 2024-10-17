# This module is from [WeNet](https://github.com/wenet-e2e/wenet).

# ## Citations

# ```bibtex
# @inproceedings{yao2021wenet,
#   title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
#   author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
#   booktitle={Proc. Interspeech},
#   year={2021},
#   address={Brno, Czech Republic },
#   organization={IEEE}
# }

# @article{zhang2022wenet,
#   title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
#   author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
#   journal={arXiv preprint arXiv:2203.15455},
#   year={2022}
# }
#

"""Encoder definition."""
from typing import Tuple, Optional, List, Union

import torch
import logging
import torch.nn.functional as F

from modules.wenet_extractor.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from modules.wenet_extractor.transformer.embedding import PositionalEncoding
from modules.wenet_extractor.transformer.embedding import RelPositionalEncoding
from modules.wenet_extractor.transformer.embedding import NoPositionalEncoding
from modules.wenet_extractor.transformer.subsampling import Conv2dSubsampling4
from modules.wenet_extractor.transformer.subsampling import Conv2dSubsampling6
from modules.wenet_extractor.transformer.subsampling import Conv2dSubsampling8
from modules.wenet_extractor.transformer.subsampling import LinearNoSubsampling
from modules.wenet_extractor.transformer.attention import MultiHeadedAttention
from modules.wenet_extractor.transformer.attention import (
    RelPositionMultiHeadedAttention,
)
from modules.wenet_extractor.transformer.encoder_layer import ConformerEncoderLayer

from modules.wenet_extractor.efficient_conformer.subsampling import Conv2dSubsampling2
from modules.wenet_extractor.efficient_conformer.convolution import ConvolutionModule
from modules.wenet_extractor.efficient_conformer.attention import (
    GroupedRelPositionMultiHeadedAttention,
)
from modules.wenet_extractor.efficient_conformer.encoder_layer import (
    StrideConformerEncoderLayer,
)

from modules.wenet_extractor.utils.common import get_activation
from modules.wenet_extractor.utils.mask import make_pad_mask
from modules.wenet_extractor.utils.mask import add_optional_chunk_mask


class EfficientConformerEncoder(torch.nn.Module):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = True,
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        stride_layer_idx: Optional[Union[int, List[int]]] = 3,
        stride: Optional[Union[int, List[int]]] = 2,
        group_layer_idx: Optional[Union[int, List[int], tuple]] = (0, 1, 2, 3),
        group_size: int = 3,
        stride_kernel: bool = True,
        **kwargs,
    ):
        """Construct Efficient Conformer Encoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            stride_layer_idx (list): layer id with StrideConv, start from 0
            stride (list): stride size of each StrideConv in efficient conformer
            group_layer_idx (list): layer id with GroupedAttention, start from 0
            group_size (int): group size of every GroupedAttention layer
            stride_kernel (bool): default True. True: recompute cnn kernels with stride.
        """
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d2":
            subsampling_class = Conv2dSubsampling2
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        logging.info(
            f"input_layer = {input_layer}, " f"subsampling_class = {subsampling_class}"
        )

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
        )
        self.input_layer = input_layer
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

        activation = get_activation(activation_type)
        self.num_blocks = num_blocks
        self.attention_heads = attention_heads
        self.cnn_module_kernel = cnn_module_kernel
        self.global_chunk_size = 0
        self.chunk_feature_map = 0

        # efficient conformer configs
        self.stride_layer_idx = (
            [stride_layer_idx] if type(stride_layer_idx) == int else stride_layer_idx
        )
        self.stride = [stride] if type(stride) == int else stride
        self.group_layer_idx = (
            [group_layer_idx] if type(group_layer_idx) == int else group_layer_idx
        )
        self.grouped_size = group_size  # group size of every GroupedAttention layer

        assert len(self.stride) == len(self.stride_layer_idx)
        self.cnn_module_kernels = [cnn_module_kernel]  # kernel size of each StridedConv
        for i in self.stride:
            if stride_kernel:
                self.cnn_module_kernels.append(self.cnn_module_kernels[-1] // i)
            else:
                self.cnn_module_kernels.append(self.cnn_module_kernels[-1])

        logging.info(
            f"stride_layer_idx= {self.stride_layer_idx}, "
            f"stride = {self.stride}, "
            f"cnn_module_kernel = {self.cnn_module_kernels}, "
            f"group_layer_idx = {self.group_layer_idx}, "
            f"grouped_size = {self.grouped_size}"
        )

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule

        # encoder definition
        index = 0
        layers = []
        for i in range(num_blocks):
            # self-attention module definition
            if i in self.group_layer_idx:
                encoder_selfattn_layer = GroupedRelPositionMultiHeadedAttention
                encoder_selfattn_layer_args = (
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    self.grouped_size,
                )
            else:
                if pos_enc_layer_type == "no_pos":
                    encoder_selfattn_layer = MultiHeadedAttention
                else:
                    encoder_selfattn_layer = RelPositionMultiHeadedAttention
                encoder_selfattn_layer_args = (
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                )

            # conformer module definition
            if i in self.stride_layer_idx:
                # conformer block with downsampling
                convolution_layer_args_stride = (
                    output_size,
                    self.cnn_module_kernels[index],
                    activation,
                    cnn_module_norm,
                    causal,
                    True,
                    self.stride[index],
                )
                layers.append(
                    StrideConformerEncoderLayer(
                        output_size,
                        encoder_selfattn_layer(*encoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        (
                            positionwise_layer(*positionwise_layer_args)
                            if macaron_style
                            else None
                        ),
                        (
                            convolution_layer(*convolution_layer_args_stride)
                            if use_cnn_module
                            else None
                        ),
                        torch.nn.AvgPool1d(
                            kernel_size=self.stride[index],
                            stride=self.stride[index],
                            padding=0,
                            ceil_mode=True,
                            count_include_pad=False,
                        ),  # pointwise_conv_layer
                        dropout_rate,
                        normalize_before,
                    )
                )
                index = index + 1
            else:
                # conformer block
                convolution_layer_args_normal = (
                    output_size,
                    self.cnn_module_kernels[index],
                    activation,
                    cnn_module_norm,
                    causal,
                )
                layers.append(
                    ConformerEncoderLayer(
                        output_size,
                        encoder_selfattn_layer(*encoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        (
                            positionwise_layer(*positionwise_layer_args)
                            if macaron_style
                            else None
                        ),
                        (
                            convolution_layer(*convolution_layer_args_normal)
                            if use_cnn_module
                            else None
                        ),
                        dropout_rate,
                        normalize_before,
                    )
                )

        self.encoders = torch.nn.ModuleList(layers)

    def set_global_chunk_size(self, chunk_size):
        """Used in ONNX export."""
        logging.info(f"set global chunk size: {chunk_size}, default is 0.")
        self.global_chunk_size = chunk_size
        if self.embed.subsampling_rate == 2:
            self.chunk_feature_map = 2 * self.global_chunk_size + 1
        elif self.embed.subsampling_rate == 6:
            self.chunk_feature_map = 6 * self.global_chunk_size + 5
        elif self.embed.subsampling_rate == 8:
            self.chunk_feature_map = 8 * self.global_chunk_size + 7
        else:
            self.chunk_feature_map = 4 * self.global_chunk_size + 3

    def output_size(self) -> int:
        return self._output_size

    def calculate_downsampling_factor(self, i: int) -> int:
        factor = 1
        for idx, stride_idx in enumerate(self.stride_layer_idx):
            if i > stride_idx:
                factor *= self.stride[idx]
        return factor

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.
        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
        )
        index = 0  # traverse stride
        for i, layer in enumerate(self.encoders):
            # layer return : x, mask, new_att_cache, new_cnn_cache
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            if i in self.stride_layer_idx:
                masks = masks[:, :, :: self.stride[index]]
                chunk_masks = chunk_masks[
                    :, :: self.stride[index], :: self.stride[index]
                ]
                mask_pad = masks
                pos_emb = pos_emb[:, :: self.stride[index], :]
                index = index + 1

        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            att_mask : mask matrix of self attention

        Returns:
            torch.Tensor: output of current input xs
            torch.Tensor: subsampling cache required for next chunk computation
            List[torch.Tensor]: encoder layers output cache required for next
                chunk computation
            List[torch.Tensor]: conformer cnn cache

        """
        assert xs.size(0) == 1

        # using downsampling factor to recover offset
        offset *= self.calculate_downsampling_factor(self.num_blocks + 1)

        chunk_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        chunk_masks = chunk_masks.unsqueeze(1)  # (1, 1, xs-time)

        real_len = 0
        if self.global_chunk_size > 0:
            # for ONNX decode simulation， padding xs to chunk_size
            real_len = xs.size(1)
            pad_len = self.chunk_feature_map - real_len
            xs = F.pad(xs, (0, 0, 0, pad_len), value=0.0)
            chunk_masks = F.pad(chunk_masks, (0, pad_len), value=0.0)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        xs, pos_emb, chunk_masks = self.embed(xs, chunk_masks, offset)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        # shape(pos_emb) = (b=1, chunk_size, emb_size=output_size=hidden-dim)

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []
        mask_pad = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        mask_pad = mask_pad.unsqueeze(1)  # batchPad (b=1, 1, time=chunk_size)

        if self.global_chunk_size > 0:
            # for ONNX decode simulation
            pos_emb = self.embed.position_encoding(
                offset=max(offset - cache_t1, 0), size=cache_t1 + self.global_chunk_size
            )
            att_mask[:, :, -self.global_chunk_size :] = chunk_masks
            mask_pad = chunk_masks.to(torch.bool)
        else:
            pos_emb = self.embed.position_encoding(
                offset=offset - cache_t1, size=attention_key_size
            )

        max_att_len, max_cnn_len = 0, 0  # for repeat_interleave of new_att_cache
        for i, layer in enumerate(self.encoders):
            factor = self.calculate_downsampling_factor(i)
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            # shape(new_att_cache) = [ batch, head, time2, outdim//head * 2 ]
            att_cache_trunc = 0
            if xs.size(1) + att_cache.size(2) / factor > pos_emb.size(1):
                # The time step is not divisible by the downsampling multiple
                att_cache_trunc = (
                    xs.size(1) + att_cache.size(2) // factor - pos_emb.size(1) + 1
                )
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                mask_pad=mask_pad,
                att_cache=att_cache[i : i + 1, :, ::factor, :][
                    :, :, att_cache_trunc:, :
                ],
                cnn_cache=cnn_cache[i, :, :, :] if cnn_cache.size(0) > 0 else cnn_cache,
            )

            if i in self.stride_layer_idx:
                # compute time dimension for next block
                efficient_index = self.stride_layer_idx.index(i)
                att_mask = att_mask[
                    :, :: self.stride[efficient_index], :: self.stride[efficient_index]
                ]
                mask_pad = mask_pad[
                    :, :: self.stride[efficient_index], :: self.stride[efficient_index]
                ]
                pos_emb = pos_emb[:, :: self.stride[efficient_index], :]

            # shape(new_att_cache) = [batch, head, time2, outdim]
            new_att_cache = new_att_cache[:, :, next_cache_start // factor :, :]
            # shape(new_cnn_cache) = [1, batch, outdim, cache_t2]
            new_cnn_cache = new_cnn_cache.unsqueeze(0)

            # use repeat_interleave to new_att_cache
            new_att_cache = new_att_cache.repeat_interleave(repeats=factor, dim=2)
            # padding new_cnn_cache to cnn.lorder for casual convolution
            new_cnn_cache = F.pad(
                new_cnn_cache, (self.cnn_module_kernel - 1 - new_cnn_cache.size(3), 0)
            )

            if i == 0:
                # record length for the first block as max length
                max_att_len = new_att_cache.size(2)
                max_cnn_len = new_cnn_cache.size(3)

            # update real shape of att_cache and cnn_cache
            r_att_cache.append(new_att_cache[:, :, -max_att_len:, :])
            r_cnn_cache.append(new_cnn_cache[:, :, :, -max_cnn_len:])

        if self.normalize_before:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        if self.global_chunk_size > 0 and real_len:
            chunk_real_len = (
                real_len
                // self.embed.subsampling_rate
                // self.calculate_downsampling_factor(self.num_blocks + 1)
            )
            # Keeping 1 more timestep can mitigate information leakage
            #   from the encoder caused by the padding
            xs = xs[:, : chunk_real_len + 1, :]

        return xs, r_att_cache, r_cnn_cache

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
        use_onnx=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            decoding_chunk_size (int): decoding chunk size
            num_decoding_left_chunks (int):
            use_onnx (bool): True for simulating ONNX model inference.
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)

        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        if use_onnx:
            logging.info("Simulating for ONNX runtime ...")
            att_cache: torch.Tensor = torch.zeros(
                (
                    self.num_blocks,
                    self.attention_heads,
                    required_cache_size,
                    self.output_size() // self.attention_heads * 2,
                ),
                device=xs.device,
            )
            cnn_cache: torch.Tensor = torch.zeros(
                (self.num_blocks, 1, self.output_size(), self.cnn_module_kernel - 1),
                device=xs.device,
            )
            self.set_global_chunk_size(chunk_size=decoding_chunk_size)
        else:
            logging.info("Simulating for JIT runtime ...")
            att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
            cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            logging.info(
                f"-->> frame chunk msg: cur={cur}, "
                f"end={end}, num_frames={end-cur}, "
                f"decoding_window={decoding_window}"
            )
            if use_onnx:
                att_mask: torch.Tensor = torch.ones(
                    (1, 1, required_cache_size + decoding_chunk_size),
                    dtype=torch.bool,
                    device=xs.device,
                )
                if cur == 0:
                    att_mask[:, :, :required_cache_size] = 0
            else:
                att_mask: torch.Tensor = torch.ones(
                    (0, 0, 0), dtype=torch.bool, device=xs.device
                )

            chunk_xs = xs[:, cur:end, :]
            (y, att_cache, cnn_cache) = self.forward_chunk(
                chunk_xs, offset, required_cache_size, att_cache, cnn_cache, att_mask
            )
            outputs.append(y)
            offset += y.size(1)

        ys = torch.cat(outputs, 1)
        masks = torch.ones(1, 1, ys.size(1), device=ys.device, dtype=torch.bool)
        return ys, masks
