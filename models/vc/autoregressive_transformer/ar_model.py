# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import LlamaConfig, LlamaForCausalLM
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.vc.autoregressive_transformer.global_encoder import GlobalEncoder


class AutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        input_vocab_size=1056,  # Eg: 1024 for only G2P-TTS, 32 for content tokens of only VC, 1056 (1024+32) for uni-training of TTS and VC
        output_vocab_size=8192,  # Eg: 8192 for content-style tokens of VC
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        use_global_style_encoder=False,
        cfg=None,
    ):
        super().__init__()

        if cfg is not None:
            self.cfg = cfg
            self.input_vocab_size = cfg.input_vocab_size
            self.output_vocab_size = cfg.output_vocab_size
            self.hidden_size = cfg.hidden_size
            self.intermediate_size = cfg.intermediate_size
            self.num_hidden_layers = cfg.num_hidden_layers
            self.num_attention_heads = cfg.num_attention_heads
            self.use_global_style_encoder = cfg.use_global_style_encoder
        else:
            self.input_vocab_size = input_vocab_size
            self.output_vocab_size = output_vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.use_global_style_encoder = use_global_style_encoder

        ## Global Style Encoder ##
        if self.use_global_style_encoder:
            self.global_encoder = GlobalEncoder(
                input_dim=cfg.global_style_encoder.input_dim,
                output_dim=cfg.hidden_size,
                hidden_size=cfg.global_style_encoder.hidden_size,
                num_heads=cfg.global_style_encoder.num_attention_heads,
                num_layers=cfg.global_style_encoder.num_hidden_layers,
            )

        ## LLaMA Model ##
        # Five special tokens: pad, bos, eos for both input and output
        self.pad_token_id = self.input_vocab_size + self.output_vocab_size  # 9248
        self.input_bos_token_id = self.pad_token_id + 1  # 9249
        self.input_eos_token_id = self.pad_token_id + 2  # 9250
        self.output_bos_token_id = self.pad_token_id + 3  # 9251
        self.output_eos_token_id = self.pad_token_id + 4  # 9252

        self.no_loss_label = -100

        self.config = LlamaConfig(
            vocab_size=self.input_vocab_size
            + self.output_vocab_size
            + 20,  # 20 is for other special tokens during post-training
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.output_bos_token_id,
            eos_token_id=self.output_eos_token_id,
        )

        self.model = LlamaForCausalLM(self.config)

    def forward(
        self,
        input_ids,
        input_mask,
        output_ids,
        output_mask,
        mels=None,
        mels_mask=None,
    ):
        """
        Args:
            input_ids: [B, T1]
            input_mask: [B, T1]
            output_ids: [B, T2]
            output_mask: [B, T2]
            mels: [B, T, n_mels] if is not None
            mels_mask: [B, T] if is not None
        """
        # [B, T1+2]
        input_ids, input_mask, input_label = self.padding_for_input(
            input_ids,
            input_mask,
            self.input_eos_token_id,
            self.input_bos_token_id,
            self.pad_token_id,
        )
        # [B, T2+2]
        output_ids, output_mask, output_label = self.padding_for_output(
            output_ids,
            output_mask,
            self.output_eos_token_id,
            self.output_bos_token_id,
            self.pad_token_id,
        )

        if self.use_global_style_encoder:
            # [B, T1+2, D]
            input_emb = self.model.model.embed_tokens(input_ids)
            # [B, T2+2, D]
            output_emb = self.model.model.embed_tokens(output_ids)
            # [B, D] -> [B, 1, D]
            global_style_emb = self.global_encoder(mels, mels_mask).unsqueeze(1)

            # [B, T1+T2+5, D]
            llama_input_emb = torch.cat(
                [input_emb, global_style_emb, output_emb], dim=1
            )
            # [B, T1+T2+5]
            llama_attention_mask = torch.cat(
                [
                    input_mask,
                    torch.as_tensor([1], device=input_mask.device).expand(
                        llama_input_emb.size(0), 1
                    ),  # [B, 1]
                    output_mask,
                ],
                dim=-1,
            )
            # [B, T1+T2+5]
            llama_label = torch.cat(
                [
                    input_label,
                    torch.as_tensor(
                        [self.no_loss_label], device=input_label.device
                    ).expand(
                        llama_input_emb.size(0), 1
                    ),  # [B, 1]
                    output_label,
                ],
                dim=-1,
            )

            out = self.model(
                inputs_embeds=llama_input_emb,
                attention_mask=llama_attention_mask,
                labels=llama_label,
                return_dict=True,
            )
        else:
            # [B, T1+T2+4]
            llama_input_ids = torch.cat([input_ids, output_ids], dim=-1)
            llama_attention_mask = torch.cat([input_mask, output_mask], dim=-1)
            llama_label = torch.cat([input_label, output_label], dim=-1)

            out = self.model(
                llama_input_ids,
                attention_mask=llama_attention_mask,
                labels=llama_label,
                return_dict=True,
            )

        return out

    def padding_for_input(self, input_ids, input_mask, eos_id, bos_id, pad_id):
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
            input_ids + self.output_vocab_size
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

    def padding_for_output(self, output_ids, output_mask, eos_id, bos_id, pad_id):
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
        input_ids,
        prompt_mels=None,
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
            input_ids: [1, T]
            prompt_mels: [1, T, n_mels]
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
            input_ids,
            _,
            _,
        ) = self.padding_for_input(
            input_ids,
            torch.ones_like(input_ids),
            self.input_eos_token_id,
            self.input_bos_token_id,
            self.pad_token_id,
        )

        if prompt_output_ids is not None:
            prompt_output_ids, _, _ = self.padding_for_output(
                prompt_output_ids,
                torch.ones_like(prompt_output_ids),
                self.output_eos_token_id,
                self.output_bos_token_id,
                self.pad_token_id,
            )
            prompt_output_ids = prompt_output_ids[:, :-1]  # remove the eos token

        if self.use_global_style_encoder:
            # When using global style encoder, prompt_mels is required
            assert prompt_mels is not None

            input_emb = self.model.model.embed_tokens(input_ids)
            global_style_emb = self.global_encoder(
                prompt_mels,
                torch.ones_like(prompt_mels[:, :, 0]).to(prompt_mels.device),
            ).unsqueeze(1)

            llama_input_emb = torch.cat([input_emb, global_style_emb], dim=1)

            if prompt_output_ids is not None:
                prompt_output_emb = self.model.model.embed_tokens(prompt_output_ids)
                llama_input_emb = torch.cat([llama_input_emb, prompt_output_emb], dim=1)

            input_length = llama_input_emb.shape[1]

            gen_tokens = self.model.generate(
                inputs_embeds=llama_input_emb,
                do_sample=True,
                max_length=max_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.output_eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
                min_new_tokens=min_new_tokens,
            )
        else:
            # When not using global style encoder, prompt_output_ids is required
            assert prompt_output_ids is not None

            llama_input_ids = torch.cat([input_ids, prompt_output_ids], dim=-1)
            input_length = llama_input_ids.shape[1]

            gen_tokens = self.model.generate(
                llama_input_ids,
                do_sample=True,
                max_length=max_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.output_eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
                min_new_tokens=min_new_tokens,
            )

            gen_tokens = gen_tokens[:, input_length:]

        if gen_tokens[:, 0] == self.output_bos_token_id:
            gen_tokens = gen_tokens[:, 1:]
        if gen_tokens[:, -1] == self.output_eos_token_id:
            gen_tokens = gen_tokens[:, :-1]

        return gen_tokens


if __name__ == "__main__":
    from models.vc.vevo.vevo_utils import count_parameters

    # # 740M
    # model = AutoregressiveTransformer(
    #     input_vocab_size=1024,
    #     output_vocab_size=8192,
    #     hidden_size=1920,
    #     intermediate_size=7680,
    #     num_hidden_layers=12,
    #     num_attention_heads=16,
    #     use_global_style_encoder=False,
    # )
    # print(count_parameters(model))

    # # 1.1B
    # model = AutoregressiveTransformer(
    #     input_vocab_size=33,
    #     output_vocab_size=8192,
    #     hidden_size=2048,
    #     intermediate_size=8192,
    #     num_hidden_layers=16,
    #     num_attention_heads=16,
    #     use_global_style_encoder=False,
    # )
    # print(count_parameters(model))

    model = AutoregressiveTransformer(
        input_vocab_size=1024,
        output_vocab_size=8192,
        hidden_size=2048,
        intermediate_size=3072,
        num_hidden_layers=16,
        num_attention_heads=16,
        use_global_style_encoder=False,
    )
    print(count_parameters(model))
