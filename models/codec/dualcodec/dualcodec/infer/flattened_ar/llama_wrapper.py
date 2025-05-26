# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Union
import torch
import time
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

IGNORE_ID = -100  # ignored by llama
from einops import rearrange

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig


class LLM(torch.nn.Module):
    """
    LLM Module
    """

    def __init__(
        self,
        llm: LlamaForCausalLM,
        config: LlamaConfig,
        speech_vocab_size=16384,
        initial_offset=10,
        sep_token=3,
    ):
        super().__init__()
        self.llm = llm
        self.config = config
        self.speech_vocab_size = speech_vocab_size

        self.sep_token = sep_token  # offset # the last text token is used as offset
        self.eos_id = self.config.eos_token_id
        self.initial_offset = initial_offset

    def pad_unpad_sequence(
        self,
        text_token,
        text_token_len,
        speech_token,
        speech_token_len,
        sep_token,
        eos_id,  # Argument for end-of-sequence token ID
        pad_eos=True,  # New flag for controlling EOS padding
    ):
        # Unpad the sequences using their lengths
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True
        )

        combined_tokens = []
        attention_mask_list = []
        label_list = []

        for i in range(len(speech_token)):
            # Create the combined token (text + separator + speech)
            combined_sequence = torch.cat(
                [
                    text_token[i],
                    torch.tensor([sep_token]).to(self.llm.device),
                    speech_token[i],
                ]
            )

            if pad_eos:
                # Append EOS token if pad_eos is True
                combined_sequence = torch.cat(
                    [combined_sequence, torch.tensor([eos_id]).to(self.llm.device)]
                )

            combined_tokens.append(combined_sequence)

            # Create attention mask: 1 for valid tokens (text, separator, speech, eos if applicable), 0 for padding
            combined_len = len(combined_sequence)  # Total length after concatenation
            attention_mask = torch.ones(combined_len, dtype=torch.int32).to(
                self.llm.device
            )  # All tokens are attended
            attention_mask_list.append(attention_mask)

            # Create label: mask text tokens, separator token, and padding
            text_len = len(text_token[i])
            label = torch.full_like(combined_sequence, IGNORE_ID).to(
                self.llm.device
            )  # Initialize with IGNORE_ID
            label[text_len + 1 : text_len + 1 + len(speech_token[i])] = speech_token[
                i
            ]  # Keep only the speech tokens for label

            if pad_eos:
                # Keep EOS token in label if pad_eos is True
                label[-1] = eos_id

            label_list.append(label)

        # Calculate the length of each combined input
        lm_input_len = torch.tensor(
            [len(seq) for seq in combined_tokens], dtype=torch.int32
        )

        # Pad the combined tokens sequence
        lm_input = pad_sequence(combined_tokens, batch_first=True, padding_value=0)

        # Pad the attention mask sequence to the same length as lm_input
        attention_mask = pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )  # Padding token masked

        # Pad the label sequence to match lm_input
        label = pad_sequence(
            label_list, batch_first=True, padding_value=IGNORE_ID
        )  # Padding should be ignored in labels

        return lm_input, lm_input_len, attention_mask, label

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,  # Batched input of text tokens
        text_len: torch.Tensor,  # Batched input lengths
        prompt_text: torch.Tensor,  # Batched prompt text
        prompt_text_len: torch.Tensor,  # Batched prompt text lengths
        prompt_speech_token: torch.Tensor,  # Batched speech tokens
        prompt_speech_token_len: torch.Tensor,  # Batched speech token lengths
        max_length=1000,  # Maximum length to generate
        temperature=1.0,  # Sampling temperature
        top_k=20,  # Top-k sampling
        top_p=0.9,  # Top-p (nucleus) sampling
        repeat_penalty=1.1,  # Repetition penalty
        num_beams=1,  # Number of beams (set > 1 for beam search)
        **kwargs,
    ):
        text = text + self.speech_vocab_size + self.initial_offset
        prompt_speech_token = prompt_speech_token + self.initial_offset
        # 1. Prepare the combined input: text + sep + speech (batched)
        lm_input, lm_input_len, lm_mask, _ = self.pad_unpad_sequence(
            text,
            text_len,
            prompt_speech_token,
            prompt_speech_token_len,
            self.sep_token,
            self.eos_id,
            pad_eos=False,
        )

        # 2. Ensure batched input: lm_input.shape = [batch_size, seq_len]
        # We assume the input is already batched

        # 3. Generate batched sequences using LlamaForCausalLM's generate function
        generated_ids = self.llm.generate(
            input_ids=lm_input.to(self.llm.device),  # Batched input tokens
            attention_mask=lm_mask.to(self.llm.device),  # Batched attention mask
            max_length=max_length,  # Maximum length to generate
            temperature=temperature,  # Sampling temperature
            top_k=top_k,  # Top-k sampling
            top_p=top_p,  # Top-p sampling
            repetition_penalty=repeat_penalty,  # Penalize repetitions
            num_beams=num_beams,  # Beam search or regular sampling
            do_sample=True,  # Enable sampling
            pad_token_id=self.config.pad_token_id,  # Padding token ID
            eos_token_id=self.eos_id,  # End-of-sequence token ID
        )
        return generated_ids[..., lm_input.shape[-1] : -1] - self.initial_offset

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text_token: (B, L)
            text_token_lengths: (B,)
            speech_token: (B, T)
            speech_token_lengths: (B,)
            embedding: (B,)
        """
        text_token = batch["text_token"]
        text_token = text_token + self.speech_vocab_size + self.initial_offset  # offset
        text_token_len = batch["text_token_len"]
        speech_token = batch["speech_token"] + self.initial_offset
        speech_token_len = batch["speech_token_len"]

        lm_input, lm_input_len, lm_mask, lm_label = self.pad_unpad_sequence(
            text_token,
            text_token_len,
            speech_token,
            speech_token_len,
            self.sep_token,
            self.eos_id,
        )

        # 7. run lm forward
        return_dict = self.llm(
            lm_input.to(self.llm.device),
            return_dict=True,
            attention_mask=lm_mask,
            labels=lm_label,
            output_attentions=False,
            output_hidden_states=False,
        )
        loss = return_dict.loss
        accuracy_dict = {}
        return {"loss": loss, "acc": accuracy_dict}
        # with torch.no_grad():
        #     # Reshape logits to keep `k` separate
        #     logits = rearrange(logits, '(b t) k h -> b t k h', b=B)

        #     # Reshape lm_target to match the logits structure
        #     lm_target = rearrange(lm_target, 'b (t k) -> b t k', k=self.num_heads)

        #     # Initialize a dictionary to store accuracy for each `k`
        #     accuracy_dict = {}

        #     # Iterate over each `k` and compute the accuracy separately
        #     for i in range(logits.size(2)):  # Iterate over dimension `k`
        #         logits_k = logits[:, :, i, :]  # Extract logits for the i-th dimension of `k`
        #         lm_target_k = lm_target[:, :, i]  # Extract the corresponding target for the i-th dimension of `k`

        #         # Get the predicted class (argmax) from the logits
        #         preds_k = torch.argmax(logits_k, dim=-1)  # Shape: [b, t]

        #         # Mask to ignore `IGNORE_ID` in target
        #         valid_mask = (lm_target_k != IGNORE_ID)

        #         # Compare predictions to the target
        #         correct_preds = (preds_k == lm_target_k) & valid_mask

        #         # Calculate accuracy: correct predictions / total valid predictions
        #         acc_k = correct_preds.sum().item() / valid_mask.sum().item()

        #         # Store the accuracy for this `k` in the dictionary
        #         accuracy_dict[f"acc_{i}"] = torch.tensor(acc_k)

        # logits = rearrange(logits, 'b t h -> (b t) h')
        # lm_target = rearrange(lm_target, 'b t -> (b t)')
