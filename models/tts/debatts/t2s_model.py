# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.sharedctypes import Value
from re import T
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import torch.nn as nn
import tqdm
from einops import rearrange

os.chdir("./models/tts/debatts")
import sys

sys.path.append("./models/tts/debatts")
from utils.topk_sampling import top_k_top_p_filtering
import pickle


class T2SLlama_new(nn.Module):
    def __init__(
        self,
        phone_vocab_size=1024,
        target_vocab_size=2048,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        pad_token_id=3072,
        bos_target_id=3073,
        eos_target_id=3074,
        bos_phone_id=3075,
        eos_phone_id=3076,
        bos_prompt0_id=3077,
        eos_prompt0_id=3078,
        use_lang_emb=False,
        cfg=None,
    ):
        super().__init__()

        phone_vocab_size = (
            cfg.phone_vocab_size
            if cfg is not None and hasattr(cfg, "phone_vocab_size")
            else phone_vocab_size
        )
        target_vocab_size = (
            cfg.target_vocab_size
            if cfg is not None and hasattr(cfg, "target_vocab_size")
            else target_vocab_size
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        intermediate_size = (
            cfg.intermediate_size
            if cfg is not None and hasattr(cfg, "intermediate_size")
            else intermediate_size
        )
        num_hidden_layers = (
            cfg.num_hidden_layers
            if cfg is not None and hasattr(cfg, "num_hidden_layers")
            else num_hidden_layers
        )
        num_attention_heads = (
            cfg.num_attention_heads
            if cfg is not None and hasattr(cfg, "num_attention_heads")
            else num_attention_heads
        )
        pad_token_id = (
            cfg.pad_token_id
            if cfg is not None and hasattr(cfg, "pad_token_id")
            else pad_token_id
        )
        bos_target_id = (
            cfg.bos_target_id
            if cfg is not None and hasattr(cfg, "bos_target_id")
            else bos_target_id
        )
        eos_target_id = (
            cfg.eos_target_id
            if cfg is not None and hasattr(cfg, "eos_target_id")
            else eos_target_id
        )
        bos_phone_id = (
            cfg.bos_phone_id
            if cfg is not None and hasattr(cfg, "bos_phone_id")
            else bos_phone_id
        )
        eos_phone_id = (
            cfg.eos_phone_id
            if cfg is not None and hasattr(cfg, "eos_phone_id")
            else eos_phone_id
        )
        use_lang_emb = (
            cfg.use_lang_emb
            if cfg is not None and hasattr(cfg, "use_lang_emb")
            else use_lang_emb
        )
        bos_prompt0_id = (
            cfg.bos_prompt0_id
            if cfg is not None and hasattr(cfg, "bos_prompt0_id")
            else bos_prompt0_id
        )
        eos_prompt0_id = (
            cfg.eos_prompt0_id
            if cfg is not None and hasattr(cfg, "eos_prompt0_id")
            else eos_prompt0_id
        )

        self.config = LlamaConfig(
            vocab_size=phone_vocab_size + target_vocab_size + 20,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            pad_token_id=pad_token_id,
            bos_token_id=bos_target_id,
            eos_token_id=eos_target_id,
            bos_prompt0_id=bos_prompt0_id,
            eos_prompt0_id=eos_prompt0_id,
        )
        self.phone_vocab_size = phone_vocab_size
        self.target_vocab_size = target_vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.bos_target_id = bos_target_id
        self.eos_target_id = eos_target_id
        self.bos_phone_id = bos_phone_id
        self.eos_phone_id = eos_phone_id
        self.use_lang_emb = use_lang_emb
        self.bos_prompt0_id = bos_prompt0_id
        self.eos_prompt0_id = eos_prompt0_id

        self.model = LlamaForCausalLM(self.config)

        if self.use_lang_emb:
            self.lang_emb = nn.Embedding(25, hidden_size, padding_idx=0)
            torch.nn.init.normal_(self.lang_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        prompt0_ids,
        prompt0_mask,
        phone_ids,
        phone_mask,
        target_ids,
        target_mask,
        lang_id=None,
    ):
        prompt0_ids, prompt0_mask, prompt0_label, prompt0_lang_mask = (
            self.add_phone_eos_bos_label(
                prompt0_ids,
                prompt0_mask,
                self.eos_prompt0_id,
                self.bos_prompt0_id,
                self.pad_token_id,
                label="prompt0_id",
            )
        )
        phone_ids, phone_mask, phone_label, lang_mask = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
            label="phone_id",
        )
        target_ids, target_mask, target_label = self.add_target_eos_bos_label(
            target_ids,
            target_mask,
            self.eos_target_id,
            self.bos_target_id,
            self.pad_token_id,
        )

        input_token_ids = torch.cat([prompt0_ids, phone_ids, target_ids], dim=-1)
        attention_mask = torch.cat([prompt0_mask, phone_mask, target_mask], dim=-1)

        labels = torch.cat([prompt0_label, phone_label, target_label], dim=-1)

        # lang_id: (B,); lang_mask: (B, T)
        if self.use_lang_emb:
            lang_embedding = self.lang_emb(lang_id).unsqueeze(1)  # (B, d) -> (B, 1, d)
            lang_embedding = lang_embedding * torch.cat(
                [prompt0_lang_mask, lang_mask, torch.zeros_like(target_mask)], dim=-1
            ).unsqueeze(
                -1
            )  # (B, T, d)
            input_token_embedding = self.model.model.embed_tokens(
                input_token_ids
            )  # (B, T, d)
            inputs_embeds = input_token_embedding + lang_embedding

            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        else:
            out = self.model(
                input_token_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        return out

    def add_phone_eos_bos_label(
        self, phone_ids, phone_mask, phone_eos_id, phone_bos_id, pad_token_id, label
    ):
        # phone_ids: [B, T]
        # phone_mask: [B, T]

        # add 0 in the left
        lang_mask = F.pad(phone_mask, (1, 0), value=0)
        # add 0 in the right
        lang_mask = F.pad(lang_mask, (0, 1), value=0)

        if label == "phone_id":
            phone_ids = phone_ids + self.target_vocab_size * phone_mask

        phone_ids = phone_ids * phone_mask
        """Step-by-Step Computation:

            Pad phone_ids:

            After padding: [[101, 102, 103, 0]]
            Invert and Pad phone_mask:

            Inverted mask: [[0, 0, 0]]
            Padded inverted mask: [[0, 0, 0, 1]]
            Calculate EOS Insertion:

            Multiply with phone_eos_id: [[0, 0, 0, 200]]
            Combine:

            Combined result: [[101, 102, 103, 200]]
        """
        phone_ids = F.pad(phone_ids, (0, 1), value=0) + phone_eos_id * F.pad(
            1 - phone_mask, (0, 1), value=1
        )  # make pad token eos token, add eos token at the end
        phone_mask = F.pad(phone_mask, (1, 0), value=1)  # add eos mask
        phone_ids = phone_ids * phone_mask + pad_token_id * (
            1 - phone_mask
        )  # restore pad token ids
        phone_ids = F.pad(phone_ids, (1, 0), value=phone_bos_id)  # add bos token
        phone_mask = F.pad(phone_mask, (1, 0), value=1)  # add bos mask
        phone_label = -100 * torch.ones_like(
            phone_ids
        )  # loss for entire phone is not computed (passed to llama)
        return phone_ids, phone_mask, phone_label, lang_mask

    def add_target_eos_bos_label(
        self, target_ids, target_mask, target_eos_id, target_bos_id, pad_token_id
    ):
        # target_ids: [B, T]
        # target_mask: [B, T]
        target_ids = target_ids * target_mask
        target_ids = F.pad(target_ids, (0, 1), value=0) + target_eos_id * F.pad(
            1 - target_mask, (0, 1), value=1
        )
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_ids = target_ids * target_mask + pad_token_id * (1 - target_mask)
        target_ids = F.pad(target_ids, (1, 0), value=target_bos_id)
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_label = target_ids * target_mask + (-100) * (
            1 - target_mask
        )  # loss for target is computed on unmasked tokens
        return target_ids, target_mask, target_label

    def add_phone_middle_label(
        self, prompt0_ids, prompt0_mask, eos_prompt0_id, pad_token_id
    ):
        # prompt0_ids: [B, T]
        # prompt0_mask: [B, T]

        prompt0_ids = prompt0_ids * prompt0_mask
        prompt0_ids = F.pad(prompt0_ids, (0, 1), value=0) + eos_prompt0_id * F.pad(
            1 - prompt0_mask, (0, 1), value=1
        )  # Add eos_prompt0_id at the positions transitioning to padding
        prompt0_mask = F.pad(
            prompt0_mask, (1, 0), value=1
        )  # Pad the mask for the new eos_prompt0_id
        prompt0_ids = prompt0_ids * prompt0_mask + pad_token_id * (
            1 - prompt0_mask
        )  # Restore pad tokens
        prompt0_ids = F.pad(
            prompt0_ids, (1, 0), value=eos_prompt0_id
        )  # Add eos_prompt0_id at the beginning
        prompt0_mask = F.pad(
            prompt0_mask, (1, 0), value=1
        )  # Adjust the mask for the added eos_prompt0_id
        prompt0_label = prompt0_ids * prompt0_mask + (-100) * (
            1 - prompt0_mask
        )  # Set up labels for loss computation

        return prompt0_ids, prompt0_mask, prompt0_label

    @torch.no_grad()
    def sample_hf(
        self,
        phone_ids,  # the phones of prompt and target should be concatenated together
        prompt_ids,
        prompt0_ids=None,
        max_length=100000,
        temperature=0.3,
        top_k=30,
        top_p=0.7,
        repeat_penalty=3.5,
        lang_ids=None,
    ):
        if prompt0_ids is not None:
            phone_mask = torch.ones_like(phone_ids)
            prompt_mask = torch.ones_like(prompt_ids)

            prompt_mask_prompt0 = torch.ones_like(prompt0_ids)

            # downsample = DownsampleWithMask(downsample_factor=2)
            # prompt0_ids, prompt_mask_prompt0 = downsample(prompt0_ids, prompt_mask_prompt0)

            phone_ids, _, _, _ = self.add_phone_eos_bos_label(
                phone_ids,
                phone_mask,
                self.eos_phone_id,
                self.bos_phone_id,
                self.pad_token_id,
                label="phone_id",
            )
            prompt_ids, _, _ = self.add_target_eos_bos_label(
                prompt_ids,
                prompt_mask,
                self.eos_target_id,
                self.bos_target_id,
                self.pad_token_id,
            )
            prompt_ids = prompt_ids[:, :-1]  # remove end token. Make it continue mode

            prompt0_ids, _, _ = self.add_target_eos_bos_label(
                prompt0_ids,
                prompt_mask_prompt0,
                self.eos_prompt0_id,
                self.bos_prompt0_id,
                self.pad_token_id,
            )

            input_token_ids = torch.cat([prompt0_ids, phone_ids, prompt_ids], dim=-1)
            input_length = input_token_ids.shape[1]

            if lang_ids != None and self.use_lang_emb:
                lang_ids = F.pad(F.pad(lang_ids, (1, 0), value=0), (0, 1), value=0)

                input_token_embedding = self.model.model.embed_tokens(
                    input_token_ids
                )  # (B, T, d)
                # lang_ids: [1,1,1,1,1,1,2,2,2,2] which means ['en','en','en','en','en','en','zh','zh','zh','zh']
                lang_mask = torch.ones_like(phone_ids)
                lang_mask[:, 0] = 0
                lang_mask[:, -1] = 0
                lang_embedding = torch.cat(
                    [
                        self.lang_emb(lang_ids),
                        self.lang_emb(lang_ids),
                        torch.zeros(
                            lang_ids.shape[0],
                            input_token_ids.shape[1] - lang_ids.shape[1],
                            self.hidden_size,
                        ).to(input_token_ids.device),
                    ],
                    dim=1,
                ) * torch.cat(
                    [lang_mask, torch.zeros_like(prompt_ids)], dim=-1
                ).unsqueeze(
                    -1
                )

                inputs_embeds = input_token_embedding + lang_embedding

                # if prosody_features is not None:
                #
                #     prosody_features = prosody_features.unsqueeze(1).expand(-1, inputs_embeds.size(1), -1)
                #     inputs_embeds = inputs_embeds + prosody_features

                generated_ids = self.model.generate(
                    # input wav phone token ids + text token ids
                    inputs_embeds=inputs_embeds,
                    do_sample=True,
                    max_length=max_length,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_target_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repeat_penalty,
                    min_new_tokens=50,
                )
                gen_tokens = generated_ids[:, :-1]
            else:

                input_token_embedding = self.model.model.embed_tokens(input_token_ids)

                generated_ids = self.model.generate(
                    input_token_ids,
                    do_sample=True,
                    max_length=max_length,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_target_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repeat_penalty,
                    min_new_tokens=50,
                )
                gen_tokens = generated_ids[:, input_length:-1]

            return gen_tokens

        else:
            phone_mask = torch.ones_like(phone_ids)
            prompt_mask = torch.ones_like(prompt_ids)
            phone_ids, _, _, _ = self.add_phone_eos_bos_label(
                phone_ids,
                phone_mask,
                self.eos_phone_id,
                self.bos_phone_id,
                self.pad_token_id,
                label="phone_ids",
            )
            prompt_ids, _, _ = self.add_target_eos_bos_label(
                prompt_ids,
                prompt_mask,
                self.eos_target_id,
                self.bos_target_id,
                self.pad_token_id,
            )
            prompt_ids = prompt_ids[:, :-1]  # remove end token. Make it continue mode

            input_token_ids = torch.cat([phone_ids, prompt_ids], dim=-1)
            input_length = input_token_ids.shape[1]

            if lang_ids != None and self.use_lang_emb:
                lang_ids = F.pad(F.pad(lang_ids, (1, 0), value=0), (0, 1), value=0)
                # token to vector
                input_token_embedding = self.model.model.embed_tokens(
                    input_token_ids
                )  # (B, T, d)
                # lang_ids: [1,1,1,1,1,1,2,2,2,2] which means ['en','en','en','en','en','en','zh','zh','zh','zh']
                lang_mask = torch.ones_like(phone_ids)
                lang_mask[:, 0] = 0
                lang_mask[:, -1] = 0
                lang_embedding = torch.cat(
                    [
                        self.lang_emb(lang_ids),
                        torch.zeros(
                            lang_ids.shape[0],
                            input_token_ids.shape[1] - lang_ids.shape[1],
                            self.hidden_size,
                        ).to(input_token_ids.device),
                    ],
                    dim=1,
                ) * torch.cat(
                    [lang_mask, torch.zeros_like(prompt_ids)], dim=-1
                ).unsqueeze(
                    -1
                )

                inputs_embeds = input_token_embedding + lang_embedding

                generated_ids = self.model.generate(
                    # input wav phone token ids + text token ids
                    inputs_embeds=inputs_embeds,
                    do_sample=True,
                    max_length=max_length,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_target_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repeat_penalty,
                    min_new_tokens=50,
                )
                # assert generated_ids.size(1) > input_length, f"Generated tokens length {generated_ids.size(1)} is less than input length {input_length}, generated ids is {generated_ids}"
                gen_tokens = generated_ids[:, :-1]

            else:

                input_token_embedding = self.model.model.embed_tokens(input_token_ids)
                # if prosody_features is not None:
                #
                #     prosody_features = prosody_features.unsqueeze(1).expand(-1, input_token_embedding.size(1), -1)
                #     inputs_embeds = input_token_embedding + prosody_features
                #     generated_ids = self.model.generate(
                #         inputs_embeds=inputs_embeds,
                generated_ids = self.model.generate(
                    input_token_ids,
                    do_sample=True,
                    max_length=max_length,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_target_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repeat_penalty,
                    min_new_tokens=50,
                )

            return gen_tokens


class DownsampleWithMask(nn.Module):
    def __init__(self, downsample_factor=2):
        super(DownsampleWithMask, self).__init__()
        self.downsample_factor = downsample_factor

    def forward(self, x, mask):
        # x shape: (batch_size, seq_len)
        x = x.float()
        x = x.unsqueeze(1)  # add channel dimension: (batch_size, 1, seq_len)
        x = F.avg_pool1d(
            x, kernel_size=self.downsample_factor, stride=self.downsample_factor
        )
        x = x.squeeze(
            1
        )  # remove channel dimension: (batch_size, seq_len // downsample_factor)
        x = x.long()

        # average pooling
        mask = mask.float()  # convert mask to float for pooling
        mask = mask.unsqueeze(1)  # add channel dimension: (batch_size, 1, seq_len)
        mask = F.avg_pool1d(
            mask, kernel_size=self.downsample_factor, stride=self.downsample_factor
        )
