# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# code adapted from https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/muse_maskgit_pytorch.py

import math
from random import random
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import pathlib
from pathlib import Path
import torchvision.transforms as T

from typing import Callable, Optional, List

from einops import rearrange, repeat

from beartype import beartype

from models.se.anyenhance.modules.attend import Attend

from tqdm.auto import tqdm

import pdb

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def l2norm(t):
    return F.normalize(t, dim=-1)


# tensor helpers


def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, seq), device=device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim=-1).argsort(dim=-1).float()

    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


# classes


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, (self.dim,), self.gamma, self.beta)
        # return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False),
    )


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.sinusoidal_embeddings = self.create_sinusoidal_embeddings(seq_len, dim)

    def create_sinusoidal_embeddings(self, seq_len, dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        sinusoidal_embeddings = torch.zeros(seq_len, dim)
        sinusoidal_embeddings[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_embeddings[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(sinusoidal_embeddings, requires_grad=False)

    def forward(self, x):
        self.sinusoidal_embeddings.to(x.device)
        # x: [b, n, dim]
        return x + self.sinusoidal_embeddings[: x.size(1)]


# RoPE from https://github.com/meta-llama/llama/blob/main/llama/model.py#L80
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped. shape: [n, dim_head]
        x (torch.Tensor): Target tensor for broadcasting compatibility. shape: [b, n, head, dim_head]

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.to(xq_.device)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        cross_attend=False,
        scale=8,
        flash=True,
        dropout=0.0,
    ):
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.attend = Attend(flash=flash, dropout=dropout, scale=scale)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, context=None, context_mask=None):
        assert not (exists(context) ^ self.cross_attend)

        n = x.shape[-2]
        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(freqs_cis):
            q = q.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            k = k.transpose(1, 2)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, "h 1 d -> b h 1 d", b=x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        if exists(context_mask):
            context_mask = repeat(context_mask, "b j -> b h i j", h=h, i=n)
            context_mask = F.pad(context_mask, (1, 0), value=True)

        out = self.attend(q, k, v, mask=context_mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SelfTransformerBlocks(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, flash=True):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(self, x, freqs_cis=None):
        for attn, ff in self.layers:
            x = attn(x, freqs_cis=freqs_cis) + x
            x = ff(x) + x

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        dim_out=None,
        self_cond=False,
        add_mask_id=False,
        vq_layers=1,
        use_rotary_pos_enc=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        self.use_rotary_pos_enc = use_rotary_pos_enc
        if not self.use_rotary_pos_enc:
            self.pos_emb = SinusoidalPositionalEncoding(seq_len, dim)
        else:
            # rotary position encoding
            self.freqs_cis = precompute_freqs_cis(kwargs.get("dim_head", 64), seq_len)
        self.seq_len = seq_len

        # for classifier-free guidance
        self.null_embed = nn.Parameter(torch.randn(dim))

        self.transformer_blocks = SelfTransformerBlocks(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.vq_layers = vq_layers
        self.to_logits = nn.ModuleList(
            [nn.Linear(dim, self.dim_out, bias=False) for _ in range(vq_layers)]
        )

        # optional self conditioning

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def forward_with_cond_scale(
        self, *args, cond_scale=3.0, return_embed=False, **kwargs
    ):
        if cond_scale == 1:
            return self.forward(
                *args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs
            )

        logits, embed = self.forward(
            *args, return_embed=True, cond_drop_prob=0.0, **kwargs
        )

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        *args,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        neg_logits = self.forward(
            *args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs
        )
        pos_logits, embed = self.forward(
            *args,
            return_embed=True,
            text_embed=text_embed,
            cond_drop_prob=0.0,
            **kwargs,
        )

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward(
        self,
        code_embeds: torch.Tensor,
        audio_embeds: list[torch.Tensor],
        task_embeds: Optional[torch.Tensor] = None,
        return_embed=False,
        return_logits=False,
        labels=None,
        ignore_index=0,
        self_cond_embed=None,
        cond_drop_prob=0.0,
        conditioning_token_ids: Optional[torch.Tensor] = None,
    ):
        # code_embeds: [b, n, dim]
        for i in range(len(audio_embeds)):
            assert (
                audio_embeds[i].shape == code_embeds.shape
            ), f"audio_embeds{i} shape {audio_embeds[i].shape} must match code_embeds shape {code_embeds.shape}"
        device, b, n = code_embeds.device, code_embeds.shape[0], code_embeds.shape[1]
        assert n <= self.seq_len

        audio_embeds = torch.sum(torch.stack(audio_embeds, dim=1), dim=1)  # [b, n, dim]

        # classifier free guidance
        if cond_drop_prob > 0.0:
            mask = prob_mask_like((b, 1), cond_drop_prob, device=device)
            mask = mask.expand(b, n)
            # mask with learnable null_embed
            audio_embeds[mask] = self.null_embed

        x = code_embeds + audio_embeds

        if exists(task_embeds):
            x = x + task_embeds

        # concat conditioning image token ids if needed

        if exists(conditioning_token_ids):
            conditioning_token_ids = rearrange(
                conditioning_token_ids, "b ... -> b (...)"
            )
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = torch.cat((context, cond_token_emb), dim=-2)
            context_mask = F.pad(
                context_mask, (0, conditioning_token_ids.shape[-1]), value=True
            )

        # embed tokens
        if not self.use_rotary_pos_enc:
            x = self.pos_emb(x)

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        if not self.use_rotary_pos_enc:
            embed = self.transformer_blocks(x)
        else:
            embed = self.transformer_blocks(x, freqs_cis=self.freqs_cis)

        logits = torch.stack(
            [linear(embed) for linear in self.to_logits], dim=1
        )  # [b, vq_layers, n, dim_out]

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        # labels: [b, vq_layers*n]
        labels = labels.reshape(b, self.vq_layers, n)
        loss = self._compute_cross_entropy(logits, labels, ignore_index)

        if not return_logits:
            return loss

        return loss, logits

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, ignore_index
    ):
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/solvers/musicgen.py#L212

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook = []
        for k in range(K):
            logits_k = (
                logits[:, k, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            q_ce = F.cross_entropy(logits_k, targets_k, ignore_index=ignore_index)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        # return ce, ce_per_codebook
        return ce


class MaskGitTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        assert "add_mask_id" not in kwargs
        super().__init__(*args, add_mask_id=True, **kwargs)


# classifier free guidance functions


def uniform(shape, min=0, max=1, device=None):
    return torch.rand(shape, device=device) * (max - min) + min


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


# noise schedules


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)
