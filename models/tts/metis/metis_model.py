# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from einops import rearrange

from models.tts.maskgct.maskgct_t2s import (
    gumbel_noise,
    gumbel_sample,
    top_k,
    MaskGCT_T2S,
)


class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_linear = nn.Linear(in_dim, out_dim * 4)
        self.act = nn.SiLU()
        self.out_linear = nn.Linear(out_dim * 4, out_dim)
        # gate scale
        self.gate_scale = nn.Parameter(torch.zeros(1))

        self.in_linear.weight.data.normal_(mean=0.0, std=0.02)
        self.in_linear.bias.data.zero_()
        self.out_linear.weight.data.normal_(mean=0.0, std=0.02)
        self.out_linear.bias.data.zero_()

    def forward(self, x):
        x = self.in_linear(x)
        x = self.act(x)
        x = self.out_linear(x)
        x = x * self.gate_scale
        return x


class MetisStage1(MaskGCT_T2S):
    def __init__(
        self,
        ft_type=None,  # tts, vc, se, tse, l2s, omni
        ft_cond_dim=1024,
        use_zero_gate_adapter=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if ft_type is not None and ft_type in ["vc", "se", "tse", "l2s", "omni"]:
            if use_zero_gate_adapter:
                self.cond_adapter = SimpleAdapter(ft_cond_dim, self.hidden_size)
            else:
                self.cond_adapter = nn.Sequential(
                    nn.Linear(ft_cond_dim, self.hidden_size * 4),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size * 4, self.hidden_size),
                )

        self.ft_type = ft_type

    def forward_diffusion(self, x0, t, prompt_len=None):
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
        if prompt_len is None:
            if torch.rand(1) > cfg_scale:
                prompt_len = torch.randint(
                    min(x0.shape[1] // 4, 5), int(x0.shape[1] * 0.4), (x0.shape[0],)
                ).to(
                    x0.device
                )  # (B,)
            else:
                prompt_len = torch.zeros(x0.shape[0]).to(x0)  # (B,)
        # else use inputed prompt_len

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

    def loss_t(
        self,
        x0,
        x_mask,
        t,
        phone_embedding=None,
        phone_mask=None,
        prompt_len=None,
        finetune_cond=None,
    ):
        xt, new_t, mask, prompt_len, mask_prob = self.forward_diffusion(
            x0, t, prompt_len
        )

        if finetune_cond is not None:
            if self.ft_type in ["tse", "vc", "se", "l2s", "omni"]:
                finetune_cond = self.cond_adapter(finetune_cond)
                xt = xt + finetune_cond

        embeds = self.diff_estimator(
            xt, new_t, x_mask, phone_embedding=phone_embedding, phone_mask=phone_mask
        )  # (B, T, hidden_size)
        logits = self.to_logit(embeds)  # (B, T, codebook_size)

        # final mask used for loss calculation
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        return logits, final_mask, x0, prompt_len, mask_prob

    def compute_loss(
        self,
        x0,
        x_mask,
        phone_embedding=None,
        phone_mask=None,
        prompt_len=None,
        finetune_cond=None,
    ):
        # x0: (B, T)
        # x_mask: (B, T) mask is 0 for padding
        t = torch.rand(x0.shape[0], device=x0.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        return self.loss_t(
            x0, x_mask, t, phone_embedding, phone_mask, prompt_len, finetune_cond
        )

    def forward(
        self,
        x0,
        x_mask,
        phone_id=None,
        phone_mask=None,
        prompt_len=None,
        finetune_cond=None,
    ):
        # x0: (B, T)
        # x_mask: (B, T) mask is 0 for padding

        if self.use_phone_cond and phone_id != None:
            phone_embedding = self.phone_emb(phone_id)
        else:
            phone_embedding = None

        logits, final_mask, x0, prompt_len, mask_prob = self.compute_loss(
            x0,
            x_mask,
            phone_embedding,
            phone_mask=phone_mask,
            prompt_len=prompt_len,
            finetune_cond=finetune_cond,
        )
        return logits, final_mask, x0, prompt_len, mask_prob

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
        finetune_cond=None,
        preschedule_mask_indices=None,
    ):
        # prompt: (B, T)
        if self.use_phone_cond and phone_id != None:
            phone_embedding = self.phone_emb(phone_id)
            phone_mask = torch.ones_like(phone_id)
        else:
            phone_embedding = None
            phone_mask = None

        prompt_code = prompt  # (B, prompt_len)
        prompt_len = prompt_code.shape[1]

        x_mask = torch.ones(prompt_code.shape[0], target_len).to(
            prompt_code.device
        )  # (B, target_len)

        if prompt_mask == None:
            prompt_mask = torch.ones(prompt_code.shape[0], prompt_len).to(
                prompt_code.device
            )  # (B, prompt_len)

        cum = torch.zeros(x_mask.shape[0], x_mask.shape[1], self.hidden_size).to(
            x_mask.device
        )  # (B, T, hidden_size)

        bsz, seq_len, _ = cum.shape

        if finetune_cond is not None:
            if self.ft_type in ["vc", "se", "tse", "l2s", "omni"]:
                finetune_cond = self.cond_adapter(finetune_cond)
                finetune_cond_wo_prompt = finetune_cond[:, prompt_len:, :]

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

            if finetune_cond is not None:
                if self.ft_type in ["vc", "se", "tse", "l2s", "omni"]:
                    xt_input = xt_input + finetune_cond

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
                input_cur = cur
                if finetune_cond is not None:
                    if self.ft_type in ["vc", "se", "tse", "l2s", "omni"]:
                        input_cur = input_cur + finetune_cond_wo_prompt
                mask_embeds = self.diff_estimator(
                    input_cur,
                    t,
                    x_mask,
                    phone_embedding=(
                        phone_embedding[:, phone_embedding.shape[1] :, :]
                        if phone_embedding is not None
                        else None
                    ),
                    phone_mask=(
                        phone_mask[:, prompt_len:]
                        if phone_embedding is not None
                        else None
                    ),
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

            if preschedule_mask_indices is not None:
                mask_indices = torch.LongTensor(
                    preschedule_mask_indices[:next_mask_num]
                ).to(
                    x_mask.device
                )  # [len]
                # repeat to [batch, len]
                mask_indices = mask_indices.unsqueeze(0).repeat(
                    x_mask.size(0), 1
                )  # [batch, len]
            else:
                mask_indices = scores.topk(next_mask_num, dim=-1).indices

            mask = torch.zeros_like(scores, dtype=torch.bool).scatter(
                1, mask_indices, True
            )
            seq = seq.masked_fill(mask, 0)

            mask = mask.unsqueeze(-1)

        cum = cum + cond_emb(seq)
        xt = seq

        return xt
