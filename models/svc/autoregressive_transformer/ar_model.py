# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.sharedctypes import Value
from re import T

import torchaudio
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
import tqdm
from einops import rearrange


class AutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        content_vocab_size=1024,
        style_vocab_size=512,
        content_style_vocab_size=16384,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        cfg=None,
    ):
        super().__init__()

        if cfg is not None:
            self.cfg = cfg
            self.content_vocab_size = cfg.content_vocab_size
            self.style_vocab_size = cfg.style_vocab_size
            self.content_style_vocab_size = cfg.content_style_vocab_size
            self.hidden_size = cfg.hidden_size
            self.intermediate_size = cfg.intermediate_size
            self.num_hidden_layers = cfg.num_hidden_layers
            self.num_attention_heads = cfg.num_attention_heads
        else:
            self.content_vocab_size = content_vocab_size
            self.style_vocab_size = style_vocab_size
            self.content_style_vocab_size = content_style_vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads

        ## LLaMA Model ##
        # Five special tokens: pad, bos, eos for both input and output
        self.pad_token_id = (
            self.content_vocab_size
            + self.style_vocab_size
            + self.content_style_vocab_size
        )
        self.content_bos_token_id = self.pad_token_id + 1
        self.content_eos_token_id = self.pad_token_id + 2
        self.style_bos_token_id = self.pad_token_id + 3
        self.style_eos_token_id = self.pad_token_id + 4
        self.content_style_bos_token_id = self.pad_token_id + 5
        self.content_style_eos_token_id = self.pad_token_id + 6

        self.no_loss_label = -100

        self.config = LlamaConfig(
            vocab_size=self.content_vocab_size
            + self.style_vocab_size
            + self.content_style_vocab_size
            + 20,  # 20 is for other special tokens during post-training
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.content_style_bos_token_id,
            eos_token_id=self.content_style_eos_token_id,
        )
        self.model = LlamaForCausalLM(self.config)

    def forward(
        self,
        content_ids,
        content_mask,
        style_ids,
        style_mask,
        content_style_ids,
        content_style_mask,
    ):
        """
        Args:
            content_ids: [B, T1]
            content_mask: [B, T1]
            style_ids: [B, T2] if is not None
            style_mask: [B, T2] if is not None
            content_style_ids: [B, T3]
            content_style_mask: [B, T3]
        """
        # [B, T1+2]
        content_ids, content_mask, content_label = self.padding_for_content(
            content_ids,
            content_mask,
            self.content_eos_token_id,
            self.content_bos_token_id,
            self.pad_token_id,
        )
        # [B, T3+2]
        content_style_ids, content_style_mask, content_style_label = (
            self.padding_for_content_style(
                content_style_ids,
                content_style_mask,
                self.content_style_eos_token_id,
                self.content_style_bos_token_id,
                self.pad_token_id,
            )
        )

        if style_ids is not None:
            # [B, T2+2]
            style_ids, style_mask, style_label = self.padding_for_style(
                style_ids,
                style_mask,
                self.style_eos_token_id,
                self.style_bos_token_id,
                self.pad_token_id,
            )

        # print("-" * 10, "After padding:", "-" * 10)
        # print(
        #     "content_ids: ",
        #     content_ids.shape,
        #     content_ids,
        #     "max: ",
        #     content_ids.max(),
        #     "min: ",
        #     content_ids.min(),
        # )
        # print("content_mask: ", content_mask.shape, content_mask)
        # if style_ids is not None:
        #     print(
        #         "style_ids: ",
        #         style_ids.shape,
        #         style_ids,
        #         "max: ",
        #         style_ids.max(),
        #         "min: ",
        #         style_ids.min(),
        #     )
        #     print("style_mask: ", style_mask.shape, style_mask)
        # print(
        #     "content_style_ids: ",
        #     content_style_ids.shape,
        #     content_style_ids,
        #     "max: ",
        #     content_style_ids.max(),
        #     "min: ",
        #     content_style_ids.min(),
        # )
        # print("content_style_mask: ", content_style_mask.shape, content_style_mask)

        if style_ids is not None:
            # [B, T1+T2+T3+4]
            llama_input_ids = torch.cat(
                [content_ids, style_ids, content_style_ids], dim=-1
            )
            llama_attention_mask = torch.cat(
                [content_mask, style_mask, content_style_mask], dim=-1
            )
            llama_label = torch.cat(
                [content_label, style_label, content_style_label], dim=-1
            )
        else:
            # [B, T1+T3+4]
            llama_input_ids = torch.cat([content_ids, content_style_ids], dim=-1)
            llama_attention_mask = torch.cat([content_mask, content_style_mask], dim=-1)
            llama_label = torch.cat([content_label, content_style_label], dim=-1)

        out = self.model(
            llama_input_ids,
            attention_mask=llama_attention_mask,
            labels=llama_label,
            return_dict=True,
        )

        return out

    def padding_for_content(self, input_ids, input_mask, eos_id, bos_id, pad_id):
        """
        Args:
            input_ids: [B, T]
            input_mask: [B, T], whose value is 1 for valid token and 0 for pad token
        Returns:
            input_ids: [B, T+2]
            input_mask: [B, T+2], whose value is 1 for valid token and 0 for pad token
            input_label: [B, T+2], whose value is -100 for not computing loss

        Input:
            I1, I2, ..., IN, 0, 0, 0
        Output:
            BOS, I1, I2, ..., IN, EOS, PAD, PAD, PAD
        """
        input_ids = (
            input_ids + self.style_vocab_size + self.content_style_vocab_size
        ) * input_mask  # This is just for Llama, since it uses a unified codebook for both input and output

        input_ids = F.pad(input_ids, (0, 1), value=0) + eos_id * F.pad(
            1 - input_mask, (0, 1), value=1
        )  # make pad token eos token, add eos token at the end
        input_mask = F.pad(input_mask, (1, 0), value=1)  # add eos mask

        input_ids = input_ids * input_mask + pad_id * (
            1 - input_mask
        )  # restore pad token ids
        input_ids = F.pad(input_ids, (1, 0), value=bos_id)  # add bos token
        input_mask = F.pad(input_mask, (1, 0), value=1)  # add bos mask
        input_label = self.no_loss_label * torch.ones_like(
            input_ids
        )  # loss for entire phone is not computed (passed to llama)

        return input_ids.long(), input_mask.long(), input_label.long()

    def padding_for_style(self, input_ids, input_mask, eos_id, bos_id, pad_id):
        """
        Args:
            input_ids: [B, T]
            input_mask: [B, T], whose value is 1 for valid token and 0 for pad token
        Returns:
            input_ids: [B, T+2]
            input_mask: [B, T+2], whose value is 1 for valid token and 0 for pad token
            input_label: [B, T+2], whose value is -100 for not computing loss

        Input:
            I1, I2, ..., IN, 0, 0, 0
        Output:
            BOS, I1, I2, ..., IN, EOS, PAD, PAD, PAD
        """
        input_ids = (
            input_ids + self.content_style_vocab_size
        ) * input_mask  # This is just for Llama, since it uses a unified codebook for both input and output

        input_ids = F.pad(input_ids, (0, 1), value=0) + eos_id * F.pad(
            1 - input_mask, (0, 1), value=1
        )  # make pad token eos token, add eos token at the end
        input_mask = F.pad(input_mask, (1, 0), value=1)  # add eos mask

        input_ids = input_ids * input_mask + pad_id * (
            1 - input_mask
        )  # restore pad token ids
        input_ids = F.pad(input_ids, (1, 0), value=bos_id)  # add bos token
        input_mask = F.pad(input_mask, (1, 0), value=1)  # add bos mask
        input_label = self.no_loss_label * torch.ones_like(
            input_ids
        )  # loss for entire phone is not computed (passed to llama)

        return input_ids.long(), input_mask.long(), input_label.long()

    def padding_for_content_style(
        self, output_ids, output_mask, eos_id, bos_id, pad_id
    ):
        """
        Args:
            output_ids: [B, T]
            output_mask: [B, T], whose value is 1 for valid token and 0 for pad token
        Returns:
            output_ids: [B, T+2]
            output_mask: [B, T+2], whose value is 1 for valid token and 0 for pad token
            output_label: [B, T+2], whose value is -100 for not computing loss

        Input:
            O1, O2, ..., ON, 0, 0, 0
        Output:
            BOS, O1, O2, ..., ON, EOS, PAD, PAD, PAD
        """
        output_ids = output_ids * output_mask
        output_ids = F.pad(output_ids, (0, 1), value=0) + eos_id * F.pad(
            1 - output_mask, (0, 1), value=1
        )
        output_mask = F.pad(output_mask, (1, 0), value=1)
        output_ids = output_ids * output_mask + pad_id * (1 - output_mask)
        output_ids = F.pad(output_ids, (1, 0), value=bos_id)
        output_mask = F.pad(output_mask, (1, 0), value=1)
        output_label = output_ids * output_mask + self.no_loss_label * (
            1 - output_mask
        )  # loss for target is computed on unmasked tokens
        return output_ids.long(), output_mask.long(), output_label.long()

    @torch.no_grad()
    def generate(
        self,
        input_content_ids,
        input_style_ids=None,
        prompt_output_ids=None,
        max_length=2000,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repeat_penalty=1.0,
        min_new_tokens=50,
    ):
        """
        Generate for one sample.

        Args:
            input_content_ids: [1, T]
            input_style_ids: [1, T] if is not None
            prompt_output_ids: [1, T] if is not None
            max_length: int
            temperature: float
            top_k: int
            top_p: float
            repeat_penalty: float
        Returns:
            gen_tokens: [1, T]
        """
        (
            input_content_ids,
            _,
            _,
        ) = self.padding_for_content(
            input_content_ids,
            torch.ones_like(input_content_ids),
            self.content_eos_token_id,
            self.content_bos_token_id,
            self.pad_token_id,
        )

        if input_style_ids is not None:
            input_style_ids, _, _ = self.padding_for_style(
                input_style_ids,
                torch.ones_like(input_style_ids),
                self.style_eos_token_id,
                self.style_bos_token_id,
                self.pad_token_id,
            )

        if prompt_output_ids is None:
            prompt_output_ids = torch.zeros(
                (1, 0), dtype=torch.long, device=input_content_ids.device
            )

        prompt_output_ids, _, _ = self.padding_for_content_style(
            prompt_output_ids,
            torch.ones_like(prompt_output_ids),
            self.content_style_eos_token_id,
            self.content_style_bos_token_id,
            self.pad_token_id,
        )
        prompt_output_ids = prompt_output_ids[:, :-1]  # remove the eos token

        if input_style_ids is not None:
            llama_input_ids = torch.cat(
                [input_content_ids, input_style_ids, prompt_output_ids], dim=-1
            )
        else:
            llama_input_ids = torch.cat([input_content_ids, prompt_output_ids], dim=-1)

        input_length = llama_input_ids.shape[1]

        gen_tokens = self.model.generate(
            llama_input_ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.content_style_eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
            min_new_tokens=min_new_tokens,
        )

        gen_tokens = gen_tokens[:, input_length:]

        if gen_tokens[:, 0] == self.content_style_bos_token_id:
            gen_tokens = gen_tokens[:, 1:]
        if gen_tokens[:, -1] == self.content_style_eos_token_id:
            gen_tokens = gen_tokens[:, :-1]

        return gen_tokens


if __name__ == "__main__":
    from models.vc.vevo.vevo_utils import count_parameters

    # 778M
    model = AutoregressiveTransformer(
        content_vocab_size=1024,
        style_vocab_size=512,
        content_style_vocab_size=16384,
        hidden_size=1920,
        intermediate_size=7680,
        num_hidden_layers=12,
        num_attention_heads=16,
    )
    print(count_parameters(model))
