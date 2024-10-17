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

"""Multi-Head Attention layer definition."""

import math
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from modules.wenet_extractor.transformer.attention import MultiHeadedAttention


class GroupedRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper:
        https://arxiv.org/abs/1901.02860
        https://arxiv.org/abs/2109.01163
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, group_size=3):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.group_size = group_size
        self.d_k = n_feat // n_head  # for GroupedAttention
        self.n_feat = n_feat
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k * self.group_size))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k * self.group_size))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros(
            (x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def pad4group(self, Q, K, V, P, mask, group_size: int = 3):
        """
        q: (#batch, time1, size) -> (#batch, head, time1, size/head)
        k,v: (#batch, time2, size) -> (#batch, head, time2, size/head)
        p: (#batch, time2, size)
        """
        # Compute Overflows
        overflow_Q = Q.size(2) % group_size
        overflow_KV = K.size(2) % group_size

        # if-else for ONNX export
        #   0 // 0.00000000000000001 = 0
        #   1 // 1.00000000000000001 = 1
        padding_Q = (group_size - overflow_Q) * int(
            overflow_Q // (overflow_Q + 0.00000000000000001)
        )
        padding_KV = (group_size - overflow_KV) * int(
            overflow_KV // (overflow_KV + 0.00000000000000001)
        )

        batch_size, _, seq_len_KV, _ = K.size()

        # Input Padding (B, T, D) -> (B, T + P, D)
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0.0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0.0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0.0)

        if mask is not None and mask.size(2) > 0:  # time2 > 0:
            mask = mask[:, ::group_size, ::group_size]

        Q = (
            Q.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h, self.d_k * group_size)
            .transpose(1, 2)
        )
        K = (
            K.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h, self.d_k * group_size)
            .transpose(1, 2)
        )
        V = (
            V.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h, self.d_k * group_size)
            .transpose(1, 2)
        )

        # process pos_emb
        P_batch_size = P.size(0)
        overflow_P = P.size(1) % group_size
        padding_P = group_size - overflow_P if overflow_P else 0
        P = F.pad(P, (0, 0, 0, padding_P), value=0.0)
        P = P.view(P_batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)

        return Q, K, V, P, mask, padding_Q

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        padding_q: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            padding_q : for GroupedAttention in efficent conformer

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, : scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)

        # n_feat!=h*d_k may be happened in GroupAttention
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_feat)
        )  # (batch, time1, d_model)
        if padding_q is not None:
            # for GroupedAttention in efficent conformer
            x = x[:, : x.size(1) - padding_q]

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q = self.linear_q(query)
        k = self.linear_k(key)  # (#batch, time2, size)
        v = self.linear_v(value)
        p = self.linear_pos(pos_emb)  # (#batch, time2, size)

        batch_size, seq_len_KV, _ = k.size()  # seq_len_KV = time2

        # (#batch, time2, size) -> (#batch, head, time2, size/head)
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        if cache.size(0) > 0:
            # use attention cache
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        # May be k and p does not match.  eg. time2=18+18/2=27 > mask=36/2=18
        if mask is not None and mask.size(2) > 0:
            time2 = mask.size(2)
            k = k[:, :, -time2:, :]
            v = v[:, :, -time2:, :]

        # q k v p: (batch, head, time1, d_k)
        q, k, v, p, mask, padding_q = self.pad4group(q, k, v, p, mask, self.group_size)

        # q_with_bias_u & q_with_bias_v = (batch, head, time1, d_k)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k * self.group_size
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask, padding_q), new_cache
