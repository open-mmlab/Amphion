# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from .modeling_llama import LlamaDecoderLayer

NUM_QUANTIZERS = 8  # number of quantizers in total, currently assumes first layer AR.
START_QUANTIZATION_LAYER = 1  # start quantization layer
END_QUANTIZATION_LAYER = 7  # end quantization layer


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-9, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.normal_(self.to_weight.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.to_weight.weight)
        # nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # disable automatic init

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)

        return (weight * hidden_states).to(input_dtype)


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        """Override to adaptive layer norm"""
        super().__init__(config=config, layer_idx=0)  # init attention, mlp, etc.
        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )

    # add `cond` in forward function
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


from transformers.models.llama.modeling_llama import BaseModelOutputWithPast


class MultiEmbedding(nn.Module):
    def __init__(
        self,
        first_layer_num_embeddings=16384,  # Add new parameter
        num_embeddings=1034,
        embedding_dim=1024,
        num_quantization_layers=NUM_QUANTIZERS,
    ):
        super().__init__()
        # First layer uses different vocab size
        self.embeddings = nn.ModuleList(
            [nn.Embedding(first_layer_num_embeddings, embedding_dim)]
        )

        # Other layers use original vocab size
        self.embeddings.extend(
            [
                nn.Embedding(num_embeddings, embedding_dim)
                for _ in range(num_quantization_layers - 1)
            ]
        )

        # Initialize embeddings
        for i in range(num_quantization_layers):
            self.embeddings[i].weight.data.normal_(mean=0.0, std=0.02)
        self._is_hf_initialized = True

    def forward(self, input_ids):
        """Input: [num_quant, B, T] -> Output: [B, T, H]"""
        try:
            # assert (input_ids[0] < 16384).all()
            # assert (input_ids[0] >= 0).all()
            # assert (input_ids[1:] < 4096).all()
            # assert (input_ids[1:] >= 0).all()
            num_quant, B, T = input_ids.shape
            summed_embeddings = torch.zeros(
                B, T, self.embeddings[0].embedding_dim, device=input_ids.device
            )
            for i in range(num_quant):
                summed_embeddings += self.embeddings[i](input_ids[i])
            return summed_embeddings
        except:
            breakpoint()


class LlammaNARModel(LlamaModel):
    def __init__(self, config):
        """Adding adaptive layer norm, conditional embeddings, and multi-level input embeddings to the decoder layer"""
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaNARDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )

        self.embed_cond = nn.Embedding(
            NUM_QUANTIZERS, config.hidden_size
        )  # 7 quantization layers

        for layer in self.layers:
            layer.input_layernorm = LlamaAdaptiveRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
            )
            layer.post_attention_layernorm = LlamaAdaptiveRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
            )

        self.post_init()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create noncausal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        def _expand_mask(
            mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
        ):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            )

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # [num_quant, B, T]
        cond: torch.LongTensor = None,  # index for conditional embeddings, [B]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # retrieve some shape info
        batch_size, seq_length, _ = input_ids.shape

        inputs_embeds = input_ids  # [B, T, H]
        # embed cond
        cond_embedding = self.embed_cond(cond)  # [B, H]

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cond_embedding=cond_embedding,  # using cond embed
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, cond_embedding=cond_embedding)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

# from transformers.models.llama.modeling_llama import CrossEntropyLoss
from torch.nn import CrossEntropyLoss
from easydict import EasyDict as edict


class LlamaForNARModeling(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlammaNARModel(config)

        self.lm_head = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for i in range(END_QUANTIZATION_LAYER - START_QUANTIZATION_LAYER + 1)
            ]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        cond: torch.LongTensor,  # added
        prediction_target: torch.LongTensor = None,  # added. No shifting. -100 means no loss
        input_ids: torch.LongTensor = None,  # expect an embedding, [B, T, H]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Prediction target: [B, T]"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            cond=cond,  # added
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head[cond - START_QUANTIZATION_LAYER](hidden_states)

        loss = None
        loss_fct = CrossEntropyLoss()

        if prediction_target is not None:
            # calculate loss if prediction_target is provided
            logits_tmp = logits.view(-1, logits.size(-1))
            prediction_target = prediction_target.view(-1)
            loss = loss_fct(logits_tmp, prediction_target)

        return edict(
            loss=loss,
            logits=logits,
        )


class ValleNAR(nn.Module):
    def __init__(
        self,
        phone_vocab_size=51866,
        first_layer_vocab_size=16384,  # 4 * target_vocab_size
        target_vocab_size=4096,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        pad_token_id=51866 + 1024,
        bos_target_id=68252,
        eos_target_id=68253,
        bos_prompt_id=68254,
        eos_prompt_id=68255,
        bos_phone_id=68256,
        eos_phone_id=68257,
        use_input_embeds=False,
        emb_dim=256,
    ):
        super(ValleNAR, self).__init__()
        self.config = LlamaConfig(
            vocab_size=phone_vocab_size + target_vocab_size + 10,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            pad_token_id=pad_token_id,
            bos_token_id=bos_target_id,
            eos_token_id=eos_target_id,
            use_cache=False,
            target_vocab_size=target_vocab_size,
        )
        self.phone_vocab_size = phone_vocab_size
        self.first_layer_vocab_size = first_layer_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pad_token_id = pad_token_id
        self.bos_target_id = bos_target_id
        self.eos_target_id = eos_target_id
        self.bos_prompt_id = bos_prompt_id
        self.eos_prompt_id = eos_prompt_id
        self.bos_phone_id = bos_phone_id
        self.eos_phone_id = eos_phone_id
        self.model = LlamaForNARModeling(self.config)

        self.use_input_embeds = use_input_embeds
        self.phone_embedder = nn.Embedding(
            self.phone_vocab_size + 10, hidden_size
        )  # use phone_embedder to embed all eos, bos tokens
        self.prompt_embedder = MultiEmbedding(
            first_layer_num_embeddings=self.first_layer_vocab_size,
            num_embeddings=self.target_vocab_size,
            embedding_dim=hidden_size,
            num_quantization_layers=NUM_QUANTIZERS,
        )
        self.phone_embedder.weight.data.normal_(mean=0.0, std=0.02)

        # use linear mask schedule when training
        # another option is uniform
        self.mask_layer_schedule = "uniform"

        # no input embedding is used to provide speaker information
        if self.use_input_embeds:
            self.emb_linear = nn.Linear(emb_dim, hidden_size)
            self.emb_linear.weight.data.normal_(mean=0.0, std=0.01)
            self.emb_linear.bias.data.zero_()

    def forward(
        self,
        phone_ids,
        phone_mask,
        target_ids,
        target_mask,
        target_quantization_layer=None,
        prompt_len=None,
        dropout=0.0,
    ):
        """
        phone_ids: [B, T]
        phone_mask: [B, T]
        target_ids: [8,B,T]
        target_mask: [B, T]
        dropout: rate of dropping out the target tokens
        """
        # assert (target_ids < 4096).all(), "target_ids should be less than 4096"
        phone_ids = phone_ids + self.target_vocab_size
        phone_ids = phone_ids * phone_mask + (1 - phone_mask) * self.pad_token_id
        # assert (phone_ids >= 1024).all(), "phone_ids should be greater than 1024"
        # if self.phone_vocab_size != 1:
        #     phone_ids, phone_mask, phone_label = self.add_phone_eos_bos_label(
        #         phone_ids,
        #         phone_mask,
        #         self.eos_phone_id,
        #         self.bos_phone_id,
        #         self.pad_token_id,
        #     )
        phone_label = -100 * (1 - phone_mask)
        # get phone embedding
        phone_embedding = self.phone_embedder(
            phone_ids - self.target_vocab_size
        )  # [B, T, H]

        if prompt_len is not None:
            assert (
                not self.training
            )  # vscode-remote://icoding%2B615692.icoding.baidu-int.com/ssd2/lijiaqi18/AmphionVALLEv2-main/models/tts/valle_v2/valle_inference.pynce stage fix prompt len to input
            NUM_PROMPT_TOKENS = prompt_len
        else:
            assert self.training
            # randomly select a prompt length
            assert self.training  # randomize prompt len in training
            NUM_PROMPT_TOKENS = np.random.randint(
                min(target_ids.shape[-1] // 4, 5), target_ids.shape[-1] // 2
            )
        # breakpoint()
        # extract 8-level prompts
        prompt_tokens = target_ids[:, :, :NUM_PROMPT_TOKENS]  # [Q, B, T]
        prompt_mask = torch.ones_like(prompt_tokens[0])
        prompt_label = -100 * prompt_mask
        # get prompt embedding
        prompt_embedding = self.prompt_embedder(prompt_tokens)  # [B, T, H]

        # randomly select a target qnt layer to predict
        # total quant layer is 0 to 7
        if target_quantization_layer is None:
            if self.mask_layer_schedule == "linear":
                weights = torch.tensor(
                    [
                        NUM_QUANTIZERS - i
                        for i in range(
                            START_QUANTIZATION_LAYER, END_QUANTIZATION_LAYER + 1
                        )
                    ]
                )
                weights = weights / weights.sum()
                mask_layer = (
                    torch.multinomial(weights, 1, replacement=True)
                    + START_QUANTIZATION_LAYER
                )
                assert (
                    mask_layer >= START_QUANTIZATION_LAYER
                    and mask_layer <= END_QUANTIZATION_LAYER
                )
                target_quantization_layer = mask_layer.item()
            elif self.mask_layer_schedule == "cosine":
                weights = torch.tensor(
                    [
                        np.cos(i / NUM_QUANTIZERS * np.pi / 2)
                        for i in range(
                            START_QUANTIZATION_LAYER, END_QUANTIZATION_LAYER + 1
                        )
                    ]
                )
                weights = weights / weights.sum()
                mask_layer = (
                    torch.multinomial(weights, 1, replacement=True)
                    + START_QUANTIZATION_LAYER
                )
                assert (
                    mask_layer >= START_QUANTIZATION_LAYER
                    and mask_layer <= END_QUANTIZATION_LAYER
                )
                target_quantization_layer = mask_layer.item()
                breakpoint()
            elif self.mask_layer_schedule == "uniform":
                target_quantization_layer = np.random.randint(
                    START_QUANTIZATION_LAYER, END_QUANTIZATION_LAYER + 1
                )

            # print(f'target layer: {target_quantization_layer}')
        # prompt of the target part
        target_prompt_ids = target_ids[
            :target_quantization_layer, :, NUM_PROMPT_TOKENS:
        ]

        def randomly_set_elements(tensor, fraction, value):
            """
            Randomly set a fraction of the elements in a tensor to a specific value.

            Args:
            tensor (torch.Tensor): The input tensor.
            fraction (float): The fraction of elements to set to the specified value (between 0 and 1).
            value (float or int): The value to set the elements to.

            Returns:
            torch.Tensor: The tensor with some elements set to the specified value.
            """
            # Create a mask with the same shape as the tensor
            mask = torch.rand_like(tensor, dtype=torch.float32) < fraction
            # Clone the tensor to avoid modifying the original tensor
            result_tensor = tensor.clone()
            # Set the elements where the mask is True to the specified value
            result_tensor[mask] = value
            return result_tensor

        if dropout != 0.0:
            target_prompt_ids = randomly_set_elements(
                target_prompt_ids, dropout, self.target_vocab_size
            )

        target_embedding = self.prompt_embedder(target_prompt_ids)

        # mask of the target part
        target_mask = target_mask[:, NUM_PROMPT_TOKENS:]

        target_labels = target_ids[
            target_quantization_layer, :, NUM_PROMPT_TOKENS:
        ] * target_mask + (-100 * (1 - target_mask))

        # breakpoint()
        # input embeddings
        input_embeddings = torch.cat(
            [phone_embedding, prompt_embedding, target_embedding], dim=1
        )
        input_mask = torch.cat([phone_mask, prompt_mask, target_mask], dim=1)  # [B, T]
        prediction_target = torch.cat(
            [phone_label, prompt_label, target_labels], dim=1
        )  # [B, T]

        out = self.model(
            cond=torch.tensor(
                target_quantization_layer,
                device=prediction_target.device,
                dtype=torch.long,
            ),
            input_ids=input_embeddings,
            prediction_target=prediction_target,
            attention_mask=input_mask,
            return_dict=True,
        )
        logits = out.logits[:, -target_embedding.shape[1] :, :]
        targets = prediction_target[..., -target_embedding.shape[1] :]
        top1_acc = logits.argmax(-1) == targets
        try:
            top1_acc = (top1_acc * target_mask).sum() / target_mask.sum()
        except Exception as e:
            print(e)
            breakpoint()

        top5_acc = (logits.topk(5, dim=-1).indices == targets.unsqueeze(-1)).any(-1)
        top5_acc = (top5_acc * target_mask).sum() / target_mask.sum()

        top10_acc = (logits.topk(10, dim=-1).indices == targets.unsqueeze(-1)).any(-1)
        top10_acc = (top10_acc * target_mask).sum() / target_mask.sum()

        out.target_quantization_layer = target_quantization_layer
        out.top1_acc = top1_acc
        out.top5_acc = top5_acc
        out.top10_acc = top10_acc

        return out

    def add_phone_eos_bos_label(
        self, phone_ids, phone_mask, phone_eos_id, phone_bos_id, pad_token_id
    ):
        # phone_ids: [B, T]
        # phone_mask: [B, T]

        phone_ids = phone_ids + self.target_vocab_size * phone_mask

        phone_ids = phone_ids * phone_mask
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
        return phone_ids, phone_mask, phone_label

    @torch.no_grad()
    def sample_hf(
        self,
        phone_ids,  # [B, T]
        phone_mask,
        prompt_ids,  # [8, B, T]
        first_stage_ids: torch.Tensor,  # [B, T]
        top_k=50,
        top_p=1,
        temperature=1.1,
        first_stage_ids_gt=None,  # [Q, B, T]
        first_stage_ids_gt_end_layer=None,  # 2 to 8
        use_text_prompt=False,
    ):
        """
        phone_ids: [B, T]
        prompt_ids: [8, B, T]
        first_stage_ids: [B, T] result from first quant layer. Should be continuation of prompt_ids. should not contain prompt part
        """
        phone_mask = torch.ones_like(phone_ids, dtype=torch.long)
        assert prompt_ids.shape[-1] >= 5, "prompt_ids should have at least 5 tokens"
        B, T = first_stage_ids.shape  # Get shape
        padding_value = 1  # Choose a value in range [0, 4096] as padding value
        # Construct padding tensor for last 7 layers
        padded_tensor = torch.full(
            (7, B, T), padding_value, device=first_stage_ids.device
        )
        # Expand first_stage_ids to target tensor and concatenate
        first_stage_ids = first_stage_ids.unsqueeze(0)
        target_ids = torch.cat(
            [prompt_ids, padded_tensor], dim=-1
        )  # Concatenate along last dimension

        target_mask = torch.ones_like(target_ids[0], dtype=torch.long)

        if first_stage_ids_gt is not None:
            target_ids[
                :first_stage_ids_gt_end_layer, :, -first_stage_ids_gt.shape[-1] :
            ] = first_stage_ids_gt[:first_stage_ids_gt_end_layer]

        gen_len = first_stage_ids.shape[-1]

        start_qnt_layer = 1
        if first_stage_ids_gt_end_layer is not None:
            start_qnt_layer = first_stage_ids_gt_end_layer
        for qnt_level in range(start_qnt_layer, 8):
            out = self.forward(
                phone_ids=phone_ids,
                phone_mask=phone_mask,
                target_ids=target_ids,
                target_mask=target_mask,
                target_quantization_layer=qnt_level,
                prompt_len=prompt_ids.shape[-1],
            )
            logits = out.logits
            gen_tokens = torch.argmax(logits[..., :4096], dim=-1).reshape(-1)[
                -gen_len:
            ]  # [T], generated tokens in this level

            # overwrite the target_ids with the generated tokens
            target_ids[qnt_level, :, -gen_len:] = gen_tokens

        return target_ids[:, :, -gen_len:]


def test():
    model = ValleNAR().cuda()

    phone_ids = torch.LongTensor([1, 2, 3, 4, 5]).reshape(1, -1).cuda()
    phone_mask = torch.LongTensor([1, 1, 1, 1, 1]).reshape(1, -1).cuda()
    target_ids = torch.randint(high=4096, size=(8, 1, 250), dtype=torch.long).cuda()
    target_mask = torch.ones(1, 250, dtype=torch.long).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for i in range(200):
        optimizer.zero_grad()
        out = model(
            phone_ids=phone_ids,
            phone_mask=phone_mask,
            target_ids=target_ids,
            target_mask=target_mask,
            # target_quantization_layer=1+i%6,
        )
        loss = out.loss

        loss.backward()

        optimizer.step()

        print(f"iter={i}, {loss}.")
    target_ids_short = target_ids[:, :, :240]

    model.eval()
    sampled = model.sample_hf(
        phone_ids, prompt_ids=target_ids_short, first_stage_ids=target_ids[0, :, 240:]
    )

    print(target_ids[:, :, -10:])
    print(sampled)

    print((sampled == target_ids[:, :, -10:]).all())


if __name__ == "__main__":
    test()
