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


from __future__ import print_function

import os
import sys
import copy
import math
import yaml
import logging
from typing import Tuple

import torch
import numpy as np

from wenet.transformer.embedding import NoPositionalEncoding
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import init_model
from wenet.bin.export_onnx_cpu import get_args, to_numpy, print_input_output_info


try:
    import onnx
    import onnxruntime
except ImportError:
    print("Please install onnx and onnxruntime!")
    sys.exit(1)


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class BPULayerNorm(torch.nn.Module):
    """Refactor torch.nn.LayerNorm to meet 4-D dataflow."""

    def __init__(self, module, chunk_size=8, run_on_bpu=False):
        super().__init__()
        original = copy.deepcopy(module)
        self.hidden = module.weight.size(0)
        self.chunk_size = chunk_size
        self.run_on_bpu = run_on_bpu

        if self.run_on_bpu:
            self.weight = torch.nn.Parameter(
                module.weight.reshape(1, self.hidden, 1, 1).repeat(1, 1, 1, chunk_size)
            )
            self.bias = torch.nn.Parameter(
                module.bias.reshape(1, self.hidden, 1, 1).repeat(1, 1, 1, chunk_size)
            )
            self.negtive = torch.nn.Parameter(
                torch.ones((1, self.hidden, 1, chunk_size)) * -1.0
            )
            self.eps = torch.nn.Parameter(
                torch.zeros((1, self.hidden, 1, chunk_size)) + module.eps
            )
            self.mean_conv_1 = torch.nn.Conv2d(self.hidden, 1, 1, bias=False)
            self.mean_conv_1.weight = torch.nn.Parameter(
                torch.ones(self.hidden, self.hidden, 1, 1) / (1.0 * self.hidden)
            )
            self.mean_conv_2 = torch.nn.Conv2d(self.hidden, 1, 1, bias=False)
            self.mean_conv_2.weight = torch.nn.Parameter(
                torch.ones(self.hidden, self.hidden, 1, 1) / (1.0 * self.hidden)
            )
        else:
            self.norm = module

        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, self.chunk_size, self.hidden)
        orig_out = module(random_data)
        new_out = self.forward(random_data.transpose(1, 2).unsqueeze(2))
        np.testing.assert_allclose(
            to_numpy(orig_out),
            to_numpy(new_out.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.run_on_bpu:
            u = self.mean_conv_1(x)  # (1, h, 1, c)
            numerator = x + u * self.negtive  # (1, h, 1, c)
            s = torch.pow(numerator, 2)  # (1, h, 1, c)
            s = self.mean_conv_2(s)  # (1, h, 1, c)
            denominator = torch.sqrt(s + self.eps)  # (1, h, 1, c)
            x = torch.div(numerator, denominator)  # (1, h, 1, c)
            x = x * self.weight + self.bias
        else:
            x = x.squeeze(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().unsqueeze(2)
        return x


class BPUIdentity(torch.nn.Module):
    """Refactor torch.nn.Identity().
    For inserting BPU node whose input == output.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.identity_conv = torch.nn.Conv2d(
            channels, channels, 1, groups=channels, bias=False
        )
        torch.nn.init.dirac_(self.identity_conv.weight.data, groups=channels)

        self.check_equal()

    def check_equal(self):
        random_data = torch.randn(1, self.channels, 1, 10)
        result = self.forward(random_data)
        np.testing.assert_allclose(
            to_numpy(random_data), to_numpy(result), rtol=1e-02, atol=1e-03
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity with 4-D dataflow, input == output.
        Args:
            x (torch.Tensor): (batch, in_channel, 1, time)

        Returns:
            (torch.Tensor): (batch, in_channel, 1, time).
        """
        return self.identity_conv(x)


class BPULinear(torch.nn.Module):
    """Refactor torch.nn.Linear or pointwise_conv"""

    def __init__(self, module, is_pointwise_conv=False):
        super().__init__()
        # Unchanged submodules and attributes
        original = copy.deepcopy(module)
        self.idim = module.weight.size(1)
        self.odim = module.weight.size(0)
        self.is_pointwise_conv = is_pointwise_conv

        # Modify weight & bias
        self.linear = torch.nn.Conv2d(self.idim, self.odim, 1, 1)
        if is_pointwise_conv:
            # (odim, idim, kernel=1) -> (odim, idim, 1, 1)
            self.linear.weight = torch.nn.Parameter(module.weight.unsqueeze(-1))
        else:
            # (odim, idim) -> (odim, idim, 1, 1)
            self.linear.weight = torch.nn.Parameter(
                module.weight.unsqueeze(2).unsqueeze(3)
            )
        self.linear.bias = module.bias

        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 8, self.idim)
        if self.is_pointwise_conv:
            random_data = random_data.transpose(1, 2)
        original_result = module(random_data)
        if self.is_pointwise_conv:
            random_data = random_data.transpose(1, 2)
            original_result = original_result.transpose(1, 2)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        new_result = self.forward(random_data)
        np.testing.assert_allclose(
            to_numpy(original_result),
            to_numpy(new_result.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear with 4-D dataflow.
        Args:
            x (torch.Tensor): (batch, in_channel, 1, time)
        Returns:
            (torch.Tensor): (batch, out_channel, 1, time).
        """
        return self.linear(x)


class BPUGlobalCMVN(torch.nn.Module):
    """Refactor wenet/transformer/cmvn.py::GlobalCMVN"""

    def __init__(self, module):
        super().__init__()
        # Unchanged submodules and attributes
        self.norm_var = module.norm_var

        # NOTE(xcsong): Expand to 4-D tensor, (mel_dim) -> (1, 1, mel_dim, 1)
        self.mean = module.mean.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        self.istd = module.istd.unsqueeze(-1).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CMVN with 4-D dataflow.
        Args:
            x (torch.Tensor): (batch, 1, mel_dim, time)
        Returns:
            (torch.Tensor): normalized feature with same shape.
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


class BPUConv2dSubsampling8(torch.nn.Module):
    """Refactor wenet/transformer/subsampling.py::Conv2dSubsampling8

    NOTE(xcsong): Only support pos_enc_class == NoPositionalEncoding
    """

    def __init__(self, module):
        super().__init__()
        # Unchanged submodules and attributes
        original = copy.deepcopy(module)
        self.right_context = module.right_context
        self.subsampling_rate = module.subsampling_rate
        assert isinstance(module.pos_enc, NoPositionalEncoding)

        # 1. Modify self.conv
        # NOTE(xcsong): We change input shape from (1, 1, frames, mel_dim)
        #   to (1, 1, mel_dim, frames) for more efficient computation.
        self.conv = module.conv
        for idx in [0, 2, 4]:
            self.conv[idx].weight = torch.nn.Parameter(
                module.conv[idx].weight.transpose(2, 3)
            )

        # 2. Modify self.linear
        # NOTE(xcsong): Split final projection to meet the requirment of
        #   maximum kernel_size (7 for XJ3)
        self.linear = torch.nn.ModuleList()
        odim = module.linear.weight.size(0)  # 512, in this case
        freq = module.linear.weight.size(1) // odim  # 4608 // 512 == 9
        self.odim, self.freq = odim, freq
        weight = module.linear.weight.reshape(
            odim, odim, freq, 1
        )  # (odim, odim * freq) -> (odim, odim, freq, 1)
        self.split_size = []
        num_split = (freq - 1) // 7 + 1  # XJ3 requires kernel_size <= 7
        slice_begin = 0
        for idx in range(num_split):
            kernel_size = min(freq, (idx + 1) * 7) - idx * 7
            conv_ele = torch.nn.Conv2d(odim, odim, (kernel_size, 1), (kernel_size, 1))
            conv_ele.weight = torch.nn.Parameter(
                weight[:, :, slice_begin : slice_begin + kernel_size, :]
            )
            conv_ele.bias = torch.nn.Parameter(torch.zeros_like(conv_ele.bias))
            self.linear.append(conv_ele)
            self.split_size.append(kernel_size)
            slice_begin += kernel_size
        self.linear[0].bias = torch.nn.Parameter(module.linear.bias)

        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 67, 80)
        mask = torch.zeros(1, 1, 67)
        original_result, _, _ = module(random_data, mask)  # (1, 8, 512)
        random_data = random_data.transpose(1, 2).unsqueeze(0)  # (1, 1, 80, 67)
        new_result = self.forward(random_data)  # (1, 512, 1, 8)
        np.testing.assert_allclose(
            to_numpy(original_result),
            to_numpy(new_result.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x with 4-D dataflow.
        Args:
            x (torch.Tensor): Input tensor (#batch, 1, mel_dim, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, odim, 1, time'),
                where time' = time // 8.
        """
        x = self.conv(x)  # (1, odim, freq, time')
        x_out = torch.zeros(x.size(0), self.odim, 1, x.size(3))
        x = torch.split(x, self.split_size, dim=2)
        for idx, (x_part, layer) in enumerate(zip(x, self.linear)):
            x_out += layer(x_part)
        return x_out


class BPUMultiHeadedAttention(torch.nn.Module):
    """Refactor wenet/transformer/attention.py::MultiHeadedAttention

    NOTE(xcsong): Only support attention_class == MultiHeadedAttention,
        we do not consider RelPositionMultiHeadedAttention currently.
    """

    def __init__(self, module, chunk_size, left_chunks):
        super().__init__()
        # Unchanged submodules and attributes
        original = copy.deepcopy(module)
        self.d_k = module.d_k
        self.h = module.h
        n_feat = self.d_k * self.h
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks
        self.time = chunk_size * (left_chunks + 1)
        self.activation = torch.nn.Softmax(dim=-1)

        # 1. Modify self.linear_x
        self.linear_q = BPULinear(module.linear_q)
        self.linear_k = BPULinear(module.linear_k)
        self.linear_v = BPULinear(module.linear_v)
        self.linear_out = BPULinear(module.linear_out)
        # 2. denom
        self.register_buffer(
            "denom", torch.full((1, self.h, 1, 1), 1.0 / math.sqrt(self.d_k))
        )

        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, self.chunk_size, self.d_k * self.h)
        mask = torch.ones((1, self.h, self.chunk_size, self.time), dtype=torch.bool)
        cache = torch.zeros(1, self.h, self.chunk_size * self.left_chunks, self.d_k * 2)
        original_out, original_cache = module(
            random_data,
            random_data,
            random_data,
            mask[:, 0, :, :],
            torch.empty(0),
            cache,
        )
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        cache = cache.reshape(
            1, self.h, self.d_k * 2, self.chunk_size * self.left_chunks
        )
        new_out, new_cache = self.forward(
            random_data, random_data, random_data, mask, cache
        )
        np.testing.assert_allclose(
            to_numpy(original_out),
            to_numpy(new_out.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            to_numpy(original_cache),
            to_numpy(new_cache.transpose(2, 3)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            q (torch.Tensor): Query tensor (#batch, size, 1, chunk_size).
            k (torch.Tensor): Key tensor (#batch, size, 1, chunk_size).
            v (torch.Tensor): Value tensor (#batch, size, 1, chunk_size).
            mask (torch.Tensor): Mask tensor,
                (#batch, head, chunk_size, cache_t + chunk_size).
            cache (torch.Tensor): Cache tensor
                (1, head, d_k * 2, cache_t),
                where `cache_t == chunk_size * left_chunks`.


        Returns:
            torch.Tensor: Output tensor (#batch, size, 1, chunk_size).
            torch.Tensor: Cache tensor
                (1, head, d_k * 2, cache_t + chunk_size)
                where `cache_t == chunk_size * left_chunks`
        """
        # 1. Forward QKV
        q = self.linear_q(q)  # (1, d, 1, c) d == size, c == chunk_size
        k = self.linear_k(k)  # (1, d, 1, c)
        v = self.linear_v(v)  # (1, d, 1, c)
        q = q.view(1, self.h, self.d_k, self.chunk_size)
        k = k.view(1, self.h, self.d_k, self.chunk_size)
        v = v.view(1, self.h, self.d_k, self.chunk_size)
        q = q.transpose(2, 3)  # (batch, head, time1, d_k)
        k_cache, v_cache = torch.split(cache, cache.size(2) // 2, dim=2)
        k = torch.cat((k_cache, k), dim=3)
        v = torch.cat((v_cache, v), dim=3)
        new_cache = torch.cat((k, v), dim=2)
        # 2. (Q^T)K
        scores = torch.matmul(q, k) * self.denom  # (#b, n_head, time1, time2)
        # 3. Forward attention
        mask = mask.eq(0)
        scores = scores.masked_fill(mask, -float("inf"))
        attn = self.activation(scores).masked_fill(mask, 0.0)
        attn = attn.transpose(2, 3)
        x = torch.matmul(v, attn)
        x = x.view(1, self.d_k * self.h, 1, self.chunk_size)
        x_out = self.linear_out(x)
        return x_out, new_cache


class BPUConvolution(torch.nn.Module):
    """Refactor wenet/transformer/convolution.py::ConvolutionModule

    NOTE(xcsong): Only suport use_layer_norm == False
    """

    def __init__(self, module):
        super().__init__()
        # Unchanged submodules and attributes
        original = copy.deepcopy(module)
        self.lorder = module.lorder
        self.use_layer_norm = False
        self.activation = module.activation
        channels = module.pointwise_conv1.weight.size(1)
        self.channels = channels
        kernel_size = module.depthwise_conv.weight.size(2)
        assert module.use_layer_norm is False

        # 1. Modify self.pointwise_conv1
        self.pointwise_conv1 = BPULinear(module.pointwise_conv1, True)

        # 2. Modify self.depthwise_conv
        self.depthwise_conv = torch.nn.Conv2d(
            channels, channels, (1, kernel_size), stride=1, groups=channels
        )
        self.depthwise_conv.weight = torch.nn.Parameter(
            module.depthwise_conv.weight.unsqueeze(-2)
        )
        self.depthwise_conv.bias = torch.nn.Parameter(module.depthwise_conv.bias)

        # 3. Modify self.norm, Only support batchnorm2d
        self.norm = torch.nn.BatchNorm2d(channels)
        self.norm.training = False
        self.norm.num_features = module.norm.num_features
        self.norm.eps = module.norm.eps
        self.norm.momentum = module.norm.momentum
        self.norm.weight = torch.nn.Parameter(module.norm.weight)
        self.norm.bias = torch.nn.Parameter(module.norm.bias)
        self.norm.running_mean = module.norm.running_mean
        self.norm.running_var = module.norm.running_var

        # 4. Modify self.pointwise_conv2
        self.pointwise_conv2 = BPULinear(module.pointwise_conv2, True)

        # 5. Identity conv, for running `concat` on BPU
        self.identity = BPUIdentity(channels)

        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 8, self.channels)
        cache = torch.zeros((1, self.channels, self.lorder))
        original_out, original_cache = module(random_data, cache=cache)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        cache = cache.unsqueeze(2)
        new_out, new_cache = self.forward(random_data, cache)
        np.testing.assert_allclose(
            to_numpy(original_out),
            to_numpy(new_out.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            to_numpy(original_cache),
            to_numpy(new_cache.squeeze(2)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(
        self, x: torch.Tensor, cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, 1, chunk_size).
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, 1, cache_t).
        Returns:
            torch.Tensor: Output tensor (#batch, channels, 1, chunk_size).
            torch.Tensor: Cache tensor (#batch, channels, 1, cache_t).
        """
        # Concat cache
        x = torch.cat((self.identity(cache), self.identity(x)), dim=3)
        new_cache = x[:, :, :, -self.lorder :]

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, 1, dim)
        x = torch.nn.functional.glu(x, dim=1)  # (b, channel, 1, dim)

        # Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))
        x = self.pointwise_conv2(x)
        return x, new_cache


class BPUFFN(torch.nn.Module):
    """Refactor wenet/transformer/positionwise_feed_forward.py::PositionwiseFeedForward"""

    def __init__(self, module):
        super().__init__()
        # Unchanged submodules and attributes
        original = copy.deepcopy(module)
        self.activation = module.activation

        # 1. Modify self.w_x
        self.w_1 = BPULinear(module.w_1)
        self.w_2 = BPULinear(module.w_2)

        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 8, self.w_1.idim)
        original_out = module(random_data)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        new_out = self.forward(random_data)
        np.testing.assert_allclose(
            to_numpy(original_out),
            to_numpy(new_out.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, D, 1, L)
        Returns:
            output tensor, (B, D, 1, L)
        """
        return self.w_2(self.activation(self.w_1(x)))


class BPUConformerEncoderLayer(torch.nn.Module):
    """Refactor wenet/transformer/encoder_layer.py::ConformerEncoderLayer"""

    def __init__(self, module, chunk_size, left_chunks, ln_run_on_bpu=False):
        super().__init__()
        # Unchanged submodules and attributes
        original = copy.deepcopy(module)
        self.size = module.size
        assert module.normalize_before is True
        assert module.concat_after is False

        # 1. Modify submodules
        self.feed_forward_macaron = BPUFFN(module.feed_forward_macaron)
        self.self_attn = BPUMultiHeadedAttention(
            module.self_attn, chunk_size, left_chunks
        )
        self.conv_module = BPUConvolution(module.conv_module)
        self.feed_forward = BPUFFN(module.feed_forward)

        # 2. Modify norms
        self.norm_ff = BPULayerNorm(module.norm_ff, chunk_size, ln_run_on_bpu)
        self.norm_mha = BPULayerNorm(module.norm_mha, chunk_size, ln_run_on_bpu)
        self.norm_ff_macron = BPULayerNorm(
            module.norm_ff_macaron, chunk_size, ln_run_on_bpu
        )
        self.norm_conv = BPULayerNorm(module.norm_conv, chunk_size, ln_run_on_bpu)
        self.norm_final = BPULayerNorm(module.norm_final, chunk_size, ln_run_on_bpu)

        # 3. 4-D ff_scale
        self.register_buffer(
            "ff_scale", torch.full((1, self.size, 1, 1), module.ff_scale)
        )

        self.check_equal(original)

    def check_equal(self, module):
        time1 = self.self_attn.chunk_size
        time2 = self.self_attn.time
        h, d_k = self.self_attn.h, self.self_attn.d_k
        random_x = torch.randn(1, time1, self.size)
        att_mask = torch.ones(1, h, time1, time2)
        att_cache = torch.zeros(1, h, time2 - time1, d_k * 2)
        cnn_cache = torch.zeros(1, self.size, self.conv_module.lorder)
        original_x, _, original_att_cache, original_cnn_cache = module(
            random_x,
            att_mask[:, 0, :, :],
            torch.empty(0),
            att_cache=att_cache,
            cnn_cache=cnn_cache,
        )
        random_x = random_x.transpose(1, 2).unsqueeze(2)
        att_cache = att_cache.reshape(1, h, d_k * 2, time2 - time1)
        cnn_cache = cnn_cache.unsqueeze(2)
        new_x, new_att_cache, new_cnn_cache = self.forward(
            random_x, att_mask, att_cache, cnn_cache
        )
        np.testing.assert_allclose(
            to_numpy(original_att_cache),
            to_numpy(new_att_cache.transpose(2, 3)),
            rtol=1e-02,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            to_numpy(original_x),
            to_numpy(new_x.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            to_numpy(original_cnn_cache),
            to_numpy(new_cnn_cache.squeeze(2)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(
        self,
        x: torch.Tensor,
        att_mask: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, size, 1, chunk_size)
            att_mask (torch.Tensor): Mask tensor for the input
                (#batch, head, chunk_size, cache_t1 + chunk_size),
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, d_k * 2, cache_t1), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, 1, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, size, 1, chunk_size).
            torch.Tensor: att_cache tensor,
                (1, head, d_k * 2, cache_t1 + chunk_size).
            torch.Tensor: cnn_cahce tensor (#batch, size, 1, cache_t2).
        """
        # 1. ffn_macaron
        residual = x
        x = self.norm_ff_macron(x)
        x = residual + self.ff_scale * self.feed_forward_macaron(x)

        # 2. attention
        residual = x
        x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, att_mask, att_cache)
        x = residual + x_att

        # 3. convolution
        residual = x
        x = self.norm_conv(x)
        x, new_cnn_cache = self.conv_module(x, cnn_cache)
        x = residual + x

        # 4. ffn
        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)

        # 5. final post-norm
        x = self.norm_final(x)

        return x, new_att_cache, new_cnn_cache


class BPUConformerEncoder(torch.nn.Module):
    """Refactor wenet/transformer/encoder.py::ConformerEncoder"""

    def __init__(self, module, chunk_size, left_chunks, ln_run_on_bpu=False):
        super().__init__()
        # Unchanged submodules and attributes
        original = copy.deepcopy(module)
        output_size = module.output_size()
        self._output_size = module.output_size()
        self.after_norm = module.after_norm
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks
        self.head = module.encoders[0].self_attn.h
        self.layers = len(module.encoders)

        # 1. Modify submodules
        self.global_cmvn = BPUGlobalCMVN(module.global_cmvn)
        self.embed = BPUConv2dSubsampling8(module.embed)
        self.encoders = torch.nn.ModuleList()
        for layer in module.encoders:
            self.encoders.append(
                BPUConformerEncoderLayer(layer, chunk_size, left_chunks, ln_run_on_bpu)
            )

        # 2. Auxiliary conv
        self.identity_cnncache = BPUIdentity(output_size)

        self.check_equal(original)

    def check_equal(self, module):
        time1 = self.encoders[0].self_attn.chunk_size
        time2 = self.encoders[0].self_attn.time
        layers = self.layers
        h, d_k = self.head, self.encoders[0].self_attn.d_k
        decoding_window = (
            (self.chunk_size - 1) * module.embed.subsampling_rate
            + module.embed.right_context
            + 1
        )
        lorder = self.encoders[0].conv_module.lorder
        random_x = torch.randn(1, decoding_window, 80)
        att_mask = torch.ones(1, h, time1, time2)
        att_cache = torch.zeros(layers, h, time2 - time1, d_k * 2)
        cnn_cache = torch.zeros(layers, 1, self._output_size, lorder)
        orig_x, orig_att_cache, orig_cnn_cache = module.forward_chunk(
            random_x,
            0,
            time2 - time1,
            att_mask=att_mask[:, 0, :, :],
            att_cache=att_cache,
            cnn_cache=cnn_cache,
        )
        random_x = random_x.unsqueeze(0)
        att_cache = att_cache.reshape(1, h * layers, d_k * 2, time2 - time1)
        cnn_cache = cnn_cache.reshape(1, self._output_size, layers, lorder)
        new_x, new_att_cache, new_cnn_cache = self.forward(
            random_x, att_cache, cnn_cache, att_mask
        )
        caches = torch.split(new_att_cache, h, dim=1)
        caches = [c.transpose(2, 3) for c in caches]
        np.testing.assert_allclose(
            to_numpy(orig_att_cache),
            to_numpy(torch.cat(caches, dim=0)),
            rtol=1e-02,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            to_numpy(orig_x),
            to_numpy(new_x.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            to_numpy(orig_cnn_cache),
            to_numpy(new_cnn_cache.transpose(0, 2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(
        self,
        xs: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, 1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (1, head * elayers, d_k * 2, cache_t1), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (1, hidden-dim, elayers, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            att_mask (torch.Tensor): Mask tensor for the input
                (#batch, head, chunk_size, cache_t1 + chunk_size),

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, hidden-dim, 1, chunk_size).
            torch.Tensor: new attention cache required for next chunk, with
                same shape as the original att_cache.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
        """
        # xs: (B, 1, time, mel_dim) -> (B, 1, mel_dim, time)
        xs = xs.transpose(2, 3)
        xs = self.global_cmvn(xs)
        # xs: (B, 1, mel_dim, time) -> (B, hidden_dim, 1, chunk_size)
        xs = self.embed(xs)

        att_cache = torch.split(att_cache, self.head, dim=1)
        cnn_cache = self.identity_cnncache(cnn_cache)
        cnn_cache = torch.split(cnn_cache, 1, dim=2)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, new_att_cache, new_cnn_cache = layer(
                xs, att_mask, att_cache=att_cache[i], cnn_cache=cnn_cache[i]
            )
            r_att_cache.append(new_att_cache[:, :, :, self.chunk_size :])
            r_cnn_cache.append(new_cnn_cache)
        r_att_cache = torch.cat(r_att_cache, dim=1)
        r_cnn_cache = self.identity_cnncache(torch.cat(r_cnn_cache, dim=2))

        xs = xs.squeeze(2).transpose(1, 2).contiguous()
        xs = self.after_norm(xs)
        # NOTE(xcsong): 4D in, 4D out to meet the requirment of CTC input.
        xs = xs.transpose(1, 2).contiguous().unsqueeze(2)  # (B, C, 1, T)

        return (xs, r_att_cache, r_cnn_cache)


class BPUCTC(torch.nn.Module):
    """Refactor wenet/transformer/ctc.py::CTC"""

    def __init__(self, module):
        super().__init__()
        # Unchanged submodules and attributes
        original = copy.deepcopy(module)
        self.idim = module.ctc_lo.weight.size(1)
        num_class = module.ctc_lo.weight.size(0)

        # 1. Modify self.ctc_lo, Split final projection to meet the
        #   requirment of maximum in/out channels (2048 for XJ3)
        self.ctc_lo = torch.nn.ModuleList()
        self.split_size = []
        num_split = (num_class - 1) // 2048 + 1
        for idx in range(num_split):
            out_channel = min(num_class, (idx + 1) * 2048) - idx * 2048
            conv_ele = torch.nn.Conv2d(self.idim, out_channel, 1, 1)
            self.ctc_lo.append(conv_ele)
            self.split_size.append(out_channel)
        orig_weight = torch.split(module.ctc_lo.weight, self.split_size, dim=0)
        orig_bias = torch.split(module.ctc_lo.bias, self.split_size, dim=0)
        for i, (w, b) in enumerate(zip(orig_weight, orig_bias)):
            w = w.unsqueeze(2).unsqueeze(3)
            self.ctc_lo[i].weight = torch.nn.Parameter(w)
            self.ctc_lo[i].bias = torch.nn.Parameter(b)

        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 100, self.idim)
        original_result = module.ctc_lo(random_data)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        new_result = self.forward(random_data)
        np.testing.assert_allclose(
            to_numpy(original_result),
            to_numpy(new_result.squeeze(2).transpose(1, 2)),
            rtol=1e-02,
            atol=1e-03,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """frame activations, without softmax.

        Args:
            Tensor x: 4d tensor (B, hidden_dim, 1, chunk_size)
        Returns:
            torch.Tensor: (B, num_class, 1, chunk_size)
        """
        out = []
        for i, layer in enumerate(self.ctc_lo):
            out.append(layer(x))
        out = torch.cat(out, dim=1)
        return out


def export_encoder(asr_model, args):
    logger.info("Stage-1: export encoder")
    decode_window, mel_dim = args.decoding_window, args.feature_size
    encoder = BPUConformerEncoder(
        asr_model.encoder,
        args.chunk_size,
        args.num_decoding_left_chunks,
        args.ln_run_on_bpu,
    )
    encoder.eval()
    encoder_outpath = os.path.join(args.output_dir, "encoder.onnx")

    logger.info("Stage-1.1: prepare inputs for encoder")
    chunk = torch.randn((1, 1, decode_window, mel_dim))
    required_cache_size = encoder.chunk_size * encoder.left_chunks
    kv_time = required_cache_size + encoder.chunk_size
    hidden, layers = encoder._output_size, len(encoder.encoders)
    head = encoder.encoders[0].self_attn.h
    d_k = hidden // head
    lorder = encoder.encoders[0].conv_module.lorder
    att_cache = torch.zeros(1, layers * head, d_k * 2, required_cache_size)
    att_mask = torch.ones((1, head, encoder.chunk_size, kv_time))
    att_mask[:, :, :, :required_cache_size] = 0
    cnn_cache = torch.zeros((1, hidden, layers, lorder))
    inputs = (chunk, att_cache, cnn_cache, att_mask)
    logger.info(
        "chunk.size(): {} att_cache.size(): {} "
        "cnn_cache.size(): {} att_mask.size(): {}".format(
            list(chunk.size()),
            list(att_cache.size()),
            list(cnn_cache.size()),
            list(att_mask.size()),
        )
    )

    logger.info("Stage-1.2: torch.onnx.export")
    # NOTE(xcsong): Below attributes will be used in
    #   onnx2horizonbin.py::generate_config()
    attributes = {}
    attributes["input_name"] = "chunk;att_cache;cnn_cache;att_mask"
    attributes["output_name"] = "output;r_att_cache;r_cnn_cache"
    attributes["input_type"] = "featuremap;featuremap;featuremap;featuremap"
    attributes["norm_type"] = "no_preprocess;no_preprocess;no_preprocess;no_preprocess"
    attributes["input_layout_train"] = "NCHW;NCHW;NCHW;NCHW"
    attributes["input_layout_rt"] = "NCHW;NCHW;NCHW;NCHW"
    attributes[
        "input_shape"
    ] = "{}x{}x{}x{};{}x{}x{}x{};{}x{}x{}x{};{}x{}x{}x{}".format(
        chunk.size(0),
        chunk.size(1),
        chunk.size(2),
        chunk.size(3),
        att_cache.size(0),
        att_cache.size(1),
        att_cache.size(2),
        att_cache.size(3),
        cnn_cache.size(0),
        cnn_cache.size(1),
        cnn_cache.size(2),
        cnn_cache.size(3),
        att_mask.size(0),
        att_mask.size(1),
        att_mask.size(2),
        att_mask.size(3),
    )
    torch.onnx.export(  # NOTE(xcsong): only support opset==11
        encoder,
        inputs,
        encoder_outpath,
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        input_names=attributes["input_name"].split(";"),
        output_names=attributes["output_name"].split(";"),
        dynamic_axes=None,
        verbose=False,
    )
    onnx_encoder = onnx.load(encoder_outpath)
    for k in vars(args):
        meta = onnx_encoder.metadata_props.add()
        meta.key, meta.value = str(k), str(getattr(args, k))
    for k in attributes:
        meta = onnx_encoder.metadata_props.add()
        meta.key, meta.value = str(k), str(attributes[k])
    onnx.checker.check_model(onnx_encoder)
    onnx.helper.printable_graph(onnx_encoder.graph)
    onnx.save(onnx_encoder, encoder_outpath)
    print_input_output_info(onnx_encoder, "onnx_encoder")
    logger.info("Export onnx_encoder, done! see {}".format(encoder_outpath))

    logger.info("Stage-1.3: check onnx_encoder and torch_encoder")
    torch_output = []
    torch_chunk, torch_att_mask = copy.deepcopy(chunk), copy.deepcopy(att_mask)
    torch_att_cache = copy.deepcopy(att_cache)
    torch_cnn_cache = copy.deepcopy(cnn_cache)
    for i in range(10):
        logger.info(
            "torch chunk-{}: {}, att_cache: {}, cnn_cache: {}"
            ", att_mask: {}".format(
                i,
                list(torch_chunk.size()),
                list(torch_att_cache.size()),
                list(torch_cnn_cache.size()),
                list(torch_att_mask.size()),
            )
        )
        torch_att_mask[:, :, :, -(encoder.chunk_size * (i + 1)) :] = 1
        out, torch_att_cache, torch_cnn_cache = encoder(
            torch_chunk, torch_att_cache, torch_cnn_cache, torch_att_mask
        )
        torch_output.append(out)
    torch_output = torch.cat(torch_output, dim=-1)

    onnx_output = []
    onnx_chunk, onnx_att_mask = to_numpy(chunk), to_numpy(att_mask)
    onnx_att_cache = to_numpy(att_cache)
    onnx_cnn_cache = to_numpy(cnn_cache)
    ort_session = onnxruntime.InferenceSession(encoder_outpath)
    input_names = [node.name for node in onnx_encoder.graph.input]
    for i in range(10):
        logger.info(
            "onnx  chunk-{}: {}, att_cache: {}, cnn_cache: {},"
            " att_mask: {}".format(
                i,
                onnx_chunk.shape,
                onnx_att_cache.shape,
                onnx_cnn_cache.shape,
                onnx_att_mask.shape,
            )
        )
        onnx_att_mask[:, :, :, -(encoder.chunk_size * (i + 1)) :] = 1
        ort_inputs = {
            "chunk": onnx_chunk,
            "att_cache": onnx_att_cache,
            "cnn_cache": onnx_cnn_cache,
            "att_mask": onnx_att_mask,
        }
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_att_cache, onnx_cnn_cache = ort_outs[1], ort_outs[2]
        onnx_output.append(ort_outs[0])
    onnx_output = np.concatenate(onnx_output, axis=-1)

    np.testing.assert_allclose(
        to_numpy(torch_output), onnx_output, rtol=1e-03, atol=1e-04
    )
    meta = ort_session.get_modelmeta()
    logger.info("custom_metadata_map={}".format(meta.custom_metadata_map))
    logger.info("Check onnx_encoder, pass!")
    return encoder, ort_session


def export_ctc(asr_model, args):
    logger.info("Stage-2: export ctc")
    ctc = BPUCTC(asr_model.ctc).eval()
    ctc_outpath = os.path.join(args.output_dir, "ctc.onnx")

    logger.info("Stage-2.1: prepare inputs for ctc")
    hidden = torch.randn((1, args.output_size, 1, args.chunk_size))

    logger.info("Stage-2.2: torch.onnx.export")
    # NOTE(xcsong): Below attributes will be used in
    #   onnx2horizonbin.py::generate_config()
    attributes = {}
    attributes["input_name"], attributes["input_type"] = "hidden", "featuremap"
    attributes["norm_type"] = "no_preprocess"
    attributes["input_layout_train"] = "NCHW"
    attributes["input_layout_rt"] = "NCHW"
    attributes["input_shape"] = "{}x{}x{}x{}".format(
        hidden.size(0),
        hidden.size(1),
        hidden.size(2),
        hidden.size(3),
    )
    torch.onnx.export(
        ctc,
        hidden,
        ctc_outpath,
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        input_names=["hidden"],
        output_names=["probs"],
        dynamic_axes=None,
        verbose=False,
    )
    onnx_ctc = onnx.load(ctc_outpath)
    for k in vars(args):
        meta = onnx_ctc.metadata_props.add()
        meta.key, meta.value = str(k), str(getattr(args, k))
    for k in attributes:
        meta = onnx_ctc.metadata_props.add()
        meta.key, meta.value = str(k), str(attributes[k])
    onnx.checker.check_model(onnx_ctc)
    onnx.helper.printable_graph(onnx_ctc.graph)
    onnx.save(onnx_ctc, ctc_outpath)
    print_input_output_info(onnx_ctc, "onnx_ctc")
    logger.info("Export onnx_ctc, done! see {}".format(ctc_outpath))

    logger.info("Stage-2.3: check onnx_ctc and torch_ctc")
    torch_output = ctc(hidden)
    ort_session = onnxruntime.InferenceSession(ctc_outpath)
    onnx_output = ort_session.run(None, {"hidden": to_numpy(hidden)})

    np.testing.assert_allclose(
        to_numpy(torch_output), onnx_output[0], rtol=1e-03, atol=1e-04
    )
    meta = ort_session.get_modelmeta()
    logger.info("custom_metadata_map={}".format(meta.custom_metadata_map))
    logger.info("Check onnx_ctc, pass!")
    return ctc, ort_session


def export_decoder(asr_model, args):
    logger.info("Currently, Decoder is not supported.")


if __name__ == "__main__":
    torch.manual_seed(777)
    args = get_args()
    args.ln_run_on_bpu = False
    # NOTE(xcsong): XJ3 BPU only support static shapes
    assert args.chunk_size > 0
    assert args.num_decoding_left_chunks > 0
    os.system("mkdir -p " + args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    print(model)

    args.feature_size = configs["input_dim"]
    args.output_size = model.encoder.output_size()
    args.decoding_window = (
        (args.chunk_size - 1) * model.encoder.embed.subsampling_rate
        + model.encoder.embed.right_context
        + 1
    )

    export_encoder(model, args)
    export_ctc(model, args)
    export_decoder(model, args)
