# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
from models.tts.maskgct.llama_nar import DiffLlamaPrefix


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class MaskGCT_T2S(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        cfg_scale=0.2,
        cond_codebook_size=8192,
        cond_dim=1024,
        cfg=None,
    ):
        super().__init__()

        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        num_layers = (
            cfg.num_layers
            if cfg is not None and hasattr(cfg, "num_layers")
            else num_layers
        )
        num_heads = (
            cfg.num_heads
            if cfg is not None and hasattr(cfg, "num_heads")
            else num_heads
        )
        cfg_scale = (
            cfg.cfg_scale
            if cfg is not None and hasattr(cfg, "cfg_scale")
            else cfg_scale
        )
        cond_codebook_size = (
            cfg.cond_codebook_size
            if cfg is not None and hasattr(cfg, "cond_codebook_size")
            else cond_codebook_size
        )
        cond_dim = (
            cfg.cond_dim if cfg is not None and hasattr(cfg, "cond_dim") else cond_dim
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.cond_codebook_size = cond_codebook_size
        self.cond_dim = cond_dim

        self.mask_emb = nn.Embedding(1, self.hidden_size)

        self.to_logit = nn.Linear(self.hidden_size, self.cond_codebook_size)

        self.cond_emb = nn.Embedding(cond_codebook_size, self.hidden_size)

        self.phone_emb = nn.Embedding(1024, hidden_size, padding_idx=1023)

        self.reset_parameters()

        self.diff_estimator = DiffLlamaPrefix(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def mask_prob(self, t):
        return torch.sin(t * np.pi / 2).to(t.device)

    def forward_diffusion(self, x0, t):
        # x0: semantic tokens (B, T)
        new_t = t
        mask_prob = self.mask_prob(new_t)  # (B,)
        # if mask_prob[i] < 0.2, mask_prob[i] = 0.2
        mask_prob = torch.where(
            mask_prob < 0.2, torch.ones_like(mask_prob) * 0.2, mask_prob
        )
        mask_token = self.mask_emb(
            torch.LongTensor([0]).to(x0.device)
        )  # (1, hidden_size)

        xt = torch.zeros(x0.shape[0], x0.shape[1], self.hidden_size).to(x0.device)

        cfg_scale = self.cfg_scale

        #  a segment of r% sequence length is masked, where r ~ U[60, 100]
        if torch.rand(1) > cfg_scale:
            prompt_len = torch.randint(
                min(x0.shape[1] // 4, 5), int(x0.shape[1] * 0.4), (x0.shape[0],)
            ).to(
                x0.device
            )  # (B,)
        else:
            prompt_len = torch.zeros(x0.shape[0]).to(x0)  # (B,)

        # get is prompt
        is_prompt = torch.zeros_like(x0[:, :])  # (B, T)
        col_indices = (
            torch.arange(is_prompt.shape[1])
            .repeat(is_prompt.shape[0], 1)
            .to(prompt_len)
        )  # (B, T)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # (B, T) 1 if prompt

        # Add mask
        mask = torch.bernoulli(torch.ones_like(x0[:, :]) * mask_prob[..., None])
        mask[is_prompt.bool()] = 0
        mask_num = mask[:,].sum(dim=1, keepdim=False)
        all_zero_mask = (mask_num == 0).bool()
        row_indices_to_modify = torch.nonzero(all_zero_mask)
        mask[row_indices_to_modify, prompt_len[row_indices_to_modify]] = 1
        mask = mask[..., None]  # (B, T, 1)
        xt = (
            xt + mask * mask_token[:, None, :] + (1 - mask) * self.cond_emb(x0[:, :])
        )  # (B, T, hidden_size)

        return xt, new_t, mask, prompt_len, mask_prob

    def loss_t(self, x0, x_mask, t, phone_embedding=None, phone_mask=None):
        xt, new_t, mask, prompt_len, mask_prob = self.forward_diffusion(x0, t)
        # xt: (B, T, hidden_size)
        # new_t: (B,)
        # mask: (B, T, 1)   mask if 1, not mask if 0
        # prompt_len: (B,)
        # mask_prob: (B,)

        embeds = self.diff_estimator(
            xt, new_t, x_mask, phone_embedding=phone_embedding, phone_mask=phone_mask
        )  # (B, T, hidden_size)
        logits = self.to_logit(embeds)  # (B, T, codebook_size)

        # final mask used for loss calculation
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        return logits, final_mask, x0, prompt_len, mask_prob

    def compute_loss(self, x0, x_mask, phone_embedding=None, phone_mask=None):
        # x0: (B, T)
        # x_mask: (B, T) mask is 0 for padding
        t = torch.rand(x0.shape[0], device=x0.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        return self.loss_t(x0, x_mask, t, phone_embedding, phone_mask)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    @torch.no_grad()
    def reverse_diffusion(
        self,
        prompt,
        target_len,
        phone_id,
        prompt_mask=None,
        temp=0.9,
        filter_thres=0.98,
        n_timesteps=40,
        cfg=1.0,
        rescale_cfg=1.0,
    ):
        # prompt: (B, T)
        phone_embedding = self.phone_emb(phone_id)

        prompt_code = prompt  # (B, prompt_len)
        prompt_len = prompt_code.shape[1]

        x_mask = torch.ones(prompt_code.shape[0], target_len).to(
            prompt_code.device
        )  # (B, target_len)
        phone_mask = torch.ones_like(phone_id)

        if prompt_mask == None:
            prompt_mask = torch.ones(prompt_code.shape[0], prompt_len).to(
                prompt_code.device
            )  # (B, prompt_len)

        cum = torch.zeros(x_mask.shape[0], x_mask.shape[1], self.hidden_size).to(
            x_mask.device
        )  # (B, T, hidden_size)

        bsz, seq_len, _ = cum.shape

        choice_temp = 1.0
        start_temp = temp  # temperature for sampling
        start_choice_temp = choice_temp  # temperature for choicing mask tokens

        xt = torch.LongTensor(bsz, seq_len).to(x_mask.device)

        steps = n_timesteps
        to_logit = self.to_logit
        cond_emb = self.cond_emb

        mask_token = self.mask_emb(torch.LongTensor([0]).to(xt.device))
        mask = torch.full((bsz, seq_len, 1), True).to(x_mask.device)  # (B, T, 1)
        seq = torch.full((bsz, seq_len), 0).to(x_mask.device)
        h = 1.0 / steps

        cur_prompt = 0
        cur_prompt = cur_prompt + cond_emb(prompt_code)

        t_list = [1.0 - i * h for i in range(steps)]
        t_list.append(0.0)
        for i in range(steps):
            t = t_list[i] * torch.ones(bsz).to(x_mask.device)
            token = cond_emb(seq)  # (B, T, hidden_size)
            cur = cum + mask * mask_token[:, None, :] + (~mask) * token

            xt_input = torch.cat([cur_prompt, cur], dim=1)  # (B, T, hidden_size)
            xt_mask = torch.cat(
                [prompt_mask, x_mask], dim=1
            )  # (B, T), mask is 0 for padding

            embeds = self.diff_estimator(
                xt_input,
                t,
                xt_mask,
                phone_embedding=phone_embedding,
                phone_mask=phone_mask,
            )
            embeds = embeds[:, prompt_len:, :]

            # classifier free guidance
            # phone_embedding=phone_embedding[:,phone_embedding.shape[1]:,:] means phone_embedding is None
            if cfg > 0:
                mask_embeds = self.diff_estimator(
                    cur,
                    t,
                    x_mask,
                    phone_embedding=phone_embedding[:, phone_embedding.shape[1] :, :],
                    phone_mask=phone_mask[:, prompt_len:],
                )
                pos_emb_std = embeds.std()  # std(g_cond)
                embeds = embeds + cfg * (embeds - mask_embeds)  # g_cfg
                rescale_embeds = embeds * pos_emb_std / embeds.std()  # g_final
                embeds = rescale_cfg * rescale_embeds + (1 - rescale_cfg) * embeds

            logits = to_logit(embeds)  # (B, T, codebook_size)
            annealing_scale = t_list[i]

            choice_temp = start_choice_temp * annealing_scale
            temp = start_temp * annealing_scale
            logits = top_k(logits, filter_thres)

            if i == steps - 1:
                # greedy
                if steps == 1:
                    temp = 0.2
                    sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))
                else:
                    sampled_ids = logits.argmax(dim=-1)

            else:
                # sampling
                sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))

            seq = torch.where(mask.squeeze(-1), sampled_ids, seq)

            scores = logits.softmax(dim=-1)
            scores = scores.gather(2, rearrange(sampled_ids, "b n -> b n 1"))
            scores = rearrange(scores, "b n 1 -> b n")

            scores = choice_temp * gumbel_noise(scores) + scores
            scores = 1 - scores

            next_t = t_list[i + 1] * torch.ones(bsz).to(x_mask.device)

            next_mask_num = (self.mask_prob(next_t) * seq_len).long()[0].item()

            if next_mask_num == 0:
                break
            scores = scores.masked_fill(
                ~mask.squeeze(-1), -torch.finfo(scores.dtype).max
            )

            mask_indices = scores.topk(next_mask_num, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter(
                1, mask_indices, True
            )
            seq = seq.masked_fill(mask, 0)

            mask = mask.unsqueeze(-1)

        cum = cum + cond_emb(seq)
        xt = seq

        return xt

    def forward(self, x0, x_mask, phone_id=None, phone_mask=None):
        # x0: (B, T)
        # x_mask: (B, T) mask is 0 for padding

        phone_embedding = self.phone_emb(phone_id)

        logits, final_mask, x0, prompt_len, mask_prob = self.compute_loss(
            x0, x_mask, phone_embedding, phone_mask=phone_mask
        )
        return logits, final_mask, x0, prompt_len, mask_prob
