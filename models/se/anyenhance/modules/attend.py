# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# code borrowed from https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/attend.py

from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from memory_efficient_attention_pytorch.flash_attention import FlashAttentionFunction

# constants

AttentionConfig = namedtuple(
    "AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)

# helpers


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(self, scale=8, dropout=0.0, flash=False):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cuda_config = None
        self.no_hardware_detected = False

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = AttentionConfig(False, True, False)

    def flash_attn(self, q, k, v, mask=None):
        default_scale = q.shape[-1] ** -0.5

        is_cuda = q.is_cuda

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # scaled_dot_product_attention does not allow for custom scale
        # so hack it in, to support rmsnorm-ed queries and keys

        rescale = self.scale / default_scale

        q = q * (rescale**0.5)
        k = k * (rescale**0.5)

        # use naive implementation if not correct hardware

        # the below logic can also incorporate whether masking is needed or not

        use_naive = not is_cuda or not exists(self.cuda_config)

        if not is_cuda or self.no_hardware_detected:
            return FlashAttentionFunction.apply(q, k, v, mask, False, 512, 512)

        # use naive implementation
        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        try:
            raise Exception()
            with torch.backends.cuda.sdp_kernel(**self.cuda_config._asdict()):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        except:
            print_once(
                "no hardware detected, falling back to naive implementation from memory-efficient-attention-pytorch library"
            )
            self.no_hardware_detected = True

            out = FlashAttentionFunction.apply(q, k, v, mask, False, 512, 512)

        return out

    def forward(self, q, k, v, mask=None, force_non_flash=False):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        if self.flash and not force_non_flash:
            return self.flash_attn(q, k, v, mask=mask)

        # similarity

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # masking

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
