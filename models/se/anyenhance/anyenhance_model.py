# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# code adapted from https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/muse_maskgit_pytorch.py

import math
from random import random
from random import choice as random_choice
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
import torchvision.transforms as T

from typing import Callable, Optional, List

from einops import rearrange, repeat

from beartype import beartype

from tqdm.auto import tqdm

import pdb
from models.se.anyenhance.modules.anyenhance_modules import *
import torchaudio


class AudioEncoder_v2(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        input_dim,
        n_fft,
        hop_length,
        win_length,
        mlp_layers,
        mlp_activation=nn.ReLU(),
        transformer_layers=6,
        transformer_dim=512,
        transformer_heads=8,
        transformer_ff_mult=4,
        transformer_dim_head=64,
        use_rotary_pos_enc=False,
        num_transformer_paths=1,
        use_noisy_audio_embed=False,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.mlp_layers = mlp_layers

        # Next, a 1025-channel 1-D batch normalization layer normalizes the magnitude of each frequency bin
        self.batch_norm = nn.BatchNorm1d(input_dim)
        # Define MLP
        self.input_dim = input_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_layers[0]),
            mlp_activation,
            nn.Linear(mlp_layers[0], mlp_layers[1]),
        )
        self.transformer_paths = nn.ModuleList(
            [
                SelfTransformerBlocks(
                    dim=transformer_dim,
                    depth=transformer_layers,
                    dim_head=transformer_dim_head,
                    heads=transformer_heads,
                    ff_mult=transformer_ff_mult,
                    flash=True,
                )
                for _ in range(num_transformer_paths)
            ]
        )

        self.use_rotary_pos_enc = use_rotary_pos_enc
        if not self.use_rotary_pos_enc:
            self.pos_emb = SinusoidalPositionalEncoding(seq_len, transformer_dim)
        else:
            self.freqs_cis = precompute_freqs_cis(transformer_dim_head, seq_len)

        self.embed_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Linear(transformer_dim, transformer_dim),
                )
                for _ in range(num_transformer_paths)
            ]
        )
        self.use_noisy_audio_embed = use_noisy_audio_embed

    def init_noisy_embed_provider(self, device):
        from models.se.anyenhance.modules.encoder_loss import SemanticLoss

        self.noisy_embed_provider = SemanticLoss(device=device)
        self.noisy_embed_provider.requires_grad_(False)

    def forward(self, x, task_emb=None, x_spec=None):
        if self.use_noisy_audio_embed and not hasattr(self, "noisy_embed_provider"):
            self.init_noisy_embed_provider(device=x.device)

        if len(x.shape) == 3:
            x = x.squeeze(1)
        if x_spec is None:
            X = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=False,
            )  # [batch, freq, time, 2]
        else:
            X = x_spec

        X_real = X[..., 0]
        X_imag = X[..., 1]
        X_mag = (X_real**2 + X_imag**2) ** 0.15
        X_phase = torch.atan2(X_imag, X_real)

        X_mag = self.batch_norm(X_mag)  # [batch, freq, time]
        X_compressed = torch.cat(
            (X_mag, X_phase), dim=1
        )  # [b, 2*ceil((n - win_length) / hop_length + 1), 1 + n // hop_length]
        X_compressed = X_compressed[
            :, :, :-1
        ]  # [b, 2*ceil((n - win_length) / hop_length + 1), n // hop_length]
        X_compressed = X_compressed.transpose(1, 2)

        # Apply MLP
        mlp_output = self.mlp(X_compressed)

        # Reshape to [batch_size, seq_len, feature_dim] for transformer
        mlp_output = mlp_output.view(
            x.shape[0], -1, self.dim
        )  # Adjust according to actual sizes

        if task_emb is not None:
            mlp_output = mlp_output + task_emb

        embeddings = []

        if not self.use_rotary_pos_enc:
            mlp_output = self.pos_emb(mlp_output)
            # Pass through Transformer
            for path in self.transformer_paths:
                embeddings.append(path(mlp_output))
        else:
            for path in self.transformer_paths:
                embeddings.append(path(mlp_output, freqs_cis=self.freqs_cis))

        embeddings_proj = [
            self.embed_proj[i](embeddings[i]) for i in range(len(embeddings))
        ]

        if self.use_noisy_audio_embed:
            noisy_embed = self.noisy_embed_provider.extract_and_resize_embeddings(
                x.unsqueeze(1), self.seq_len, self.dim
            )
            embeddings.append(noisy_embed)

        return embeddings, embeddings_proj


class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim_out, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        logits, embeds = self.net.forward_with_cond_scale(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(logits)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        logits, embeds = self.net.forward_with_neg_prompt(
            x, *args, return_embed=True, **kwargs
        )
        return self.to_pred(logits)

    def forward(
        self, x, *args, labels=None, ignore_index=-1, return_logits=False, **kwargs
    ):
        logits_net, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(logits_net)  # [b, vq_layers, n, 1]

        if not exists(labels):
            return logits

        logits = rearrange(logits, "... 1 -> ...")  # [b, vq_layers, n]
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        mask = labels != ignore_index
        bce_loss = bce_loss * mask
        if return_logits:
            return bce_loss.sum() / mask.sum(), logits
        else:
            return bce_loss.sum() / mask.sum()


class SelfCritic_V2(nn.Module):
    # project the embeds, not the logits
    def __init__(self, net):
        super().__init__()
        self.net = net
        # self.to_pred = nn.Linear(net.dim, 1)
        self.to_pred = nn.ModuleList(
            [nn.Linear(net.dim, 1) for _ in range(net.vq_layers)]
        )

    def forward_with_cond_scale(self, x, *args, **kwargs):
        # logits = torch.stack([linear(embed) for linear in self.to_logits], dim = 1) # [b, vq_layers, n, dim_out]
        logits, embeds = self.net.forward_with_cond_scale(
            x, *args, return_embed=True, **kwargs
        )
        return torch.stack([linear(embeds) for linear in self.to_pred], dim=1)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        logits, embeds = self.net.forward_with_neg_prompt(
            x, *args, return_embed=True, **kwargs
        )
        return torch.stack([linear(embeds) for linear in self.to_pred], dim=1)

    def forward(
        self, x, *args, labels=None, ignore_index=-1, return_logits=False, **kwargs
    ):
        logits_net, embeds = self.net(x, *args, return_embed=True, **kwargs)
        # logits = self.to_pred(embeds) # [b, vq_layers, n, 1]
        logits = torch.stack(
            [linear(embeds) for linear in self.to_pred], dim=1
        )  # [b, vq_layers, n, 1]

        if not exists(labels):
            return logits

        logits = rearrange(logits, "... 1 -> ...")  # [b, vq_layers, n]
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        mask = labels != ignore_index
        bce_loss = bce_loss * mask
        if return_logits:
            return bce_loss.sum() / mask.sum(), logits
        else:
            return bce_loss.sum() / mask.sum()


class AnyEnhance(nn.Module):
    def __init__(
        self,
        seq_len: int,
        prompt_len: int,
        vq_layers: int,
        vq_model: nn.Module,
        audio_encoder: AudioEncoder_v2,
        transformer: MaskGitTransformer,
        task_num: int,
        noise_schedule: Callable = cosine_schedule,
        cond_drop_prob=0.1,
        self_cond_prob=0.9,
        no_mask_token_prob=0.0,
        prompt_prob=0.5,
        self_critic=False,
        critic_v2=False,
        critic_loss_weight=1.0,
        prompt_bandwidth_limitation_prob=0.0,
        prompt_bandwidth_limitation_rates=[8000, 16000, 22050, 24000, 32000],
    ):
        super().__init__()

        self.seq_len = seq_len
        self.prompt_len = prompt_len
        self.prompt_prob = prompt_prob
        self.vq_layers = vq_layers

        self.vq_model = vq_model

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        assert (
            self.seq_len + self.prompt_len == self.transformer.seq_len
        ), "seq_len + prompt_len must equal transformer seq_len"
        self.self_cond = transformer.self_cond

        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        # self conditioning
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob

        self.code_emb = nn.ModuleList(
            [
                nn.Embedding(self.transformer.num_tokens + 1, self.transformer.dim)
                for _ in range(vq_layers)
            ]
        )
        self.audio_encoder = audio_encoder
        self.task_emb = nn.Embedding(task_num, self.transformer.dim)

        self.self_critic = self_critic
        if self.self_critic:
            critic_class = SelfCritic_V2 if critic_v2 else SelfCritic
            print("critic_class: ", critic_class)
            self.token_critic = critic_class(self.transformer)
            self.critic_loss_weight = critic_loss_weight

        self.prompt_bandwidth_limitation_prob = prompt_bandwidth_limitation_prob
        self.prompt_bandwidth_limitation_rates = prompt_bandwidth_limitation_rates

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    @torch.no_grad()
    @eval_decorator
    def generate_with_prompt(
        self,
        noisy_audios: torch.Tensor,
        prompt_audios: torch.Tensor,
        task_type,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        timesteps=20,  # Ideal number of steps as per MaskGIT paper
        cond_scale=1,
        force_not_use_token_critic=False,
    ):
        device = next(self.parameters()).device
        batch = noisy_audios.shape[0]
        seq_len_total = (
            self.seq_len + self.prompt_len
        )  # Total sequence length per VQ layer

        # Encode prompt_audios to get prompt_audio_codes
        prompt_audio_codes = self.encode(
            prompt_audios
        )  # Shape: [batch, vq_layers, seq_len_total]
        prompt_audio_codes = prompt_audio_codes[
            :, :, : self.prompt_len
        ]  # Only keep prompt positions

        # Initialize ids with prompt_audio_codes at prompt positions and mask_id elsewhere
        ids = torch.full(
            (batch, self.vq_layers * self.seq_len),
            self.mask_id,
            dtype=torch.long,
            device=device,
        )

        # Initialize scores tensor
        scores = torch.zeros(
            (batch, self.vq_layers * self.seq_len), dtype=torch.float32, device=device
        )

        starting_temperature = temperature
        cond_ids = None
        demask_fn = self.transformer.forward_with_cond_scale
        self_cond_embed = None

        use_token_critic = self.self_critic and not force_not_use_token_critic
        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        # Zero out the prompt sections in noisy_audios
        prompt_audio_length = self.audio_encoder.hop_length * self.prompt_len
        # zero audio in prompt_audio_length
        noisy_audios_prefix = torch.zeros(
            noisy_audios.shape[0],
            noisy_audios.shape[1],
            prompt_audio_length,
            device=noisy_audios.device,
        )
        # Concatenate the zeroed out prefix with the rest of the audio
        noisy_audios = torch.cat((noisy_audios_prefix, noisy_audios), dim=-1)

        seq_len_to_mask = (
            self.seq_len * self.vq_layers
        )  # Total tokens to mask per sample

        task_embeds = (
            self.task_emb(task_type).unsqueeze(1).repeat(1, seq_len_total, 1)
        )  # [b, n, dim]
        audio_embeds, audio_embeds_proj = self.audio_encoder(
            noisy_audios, task_embeds
        )  # list([b, n, dim]), list([b, n, dim])
        for timestep, steps_until_x0 in zip(
            torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))
        ):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len_to_mask).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices
            ids = ids.scatter(1, masked_indices, self.mask_id)
            prompted_ids = torch.cat(
                (prompt_audio_codes, ids.reshape(batch, self.vq_layers, self.seq_len)),
                dim=-1,
            )

            x = prompted_ids

            # Encode x & noisy audio
            code_embeds = torch.sum(
                torch.stack(
                    [self.code_emb[i](x[:, i, :]) for i in range(self.vq_layers)], dim=1
                ),
                dim=1,
            )  # Shape: [batch, seq_len_total, dim]

            # Generate logits and embeddings
            logits, embed = demask_fn(
                code_embeds,
                audio_embeds,
                task_embeds=task_embeds,
                self_cond_embed=self_cond_embed,
                conditioning_token_ids=cond_ids,
                cond_scale=cond_scale,
                return_embed=True,
            )

            self_cond_embed = embed if self.self_cond else None

            logits = logits[
                :, :, self.prompt_len :, :
            ]  # Remove prompt positions from logits
            logits = logits.reshape(batch, self.vq_layers * self.seq_len, -1)
            # Filter logits and sample predictions
            filtered_logits = top_k(logits, topk_filter_thres)
            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # Annealed temperature
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            if use_token_critic:
                ids_critic = torch.cat(
                    (
                        prompt_audio_codes,
                        ids.reshape(batch, self.vq_layers, self.seq_len),
                    ),
                    dim=-1,
                )
                code_embeds_critic = torch.sum(
                    torch.stack(
                        [
                            self.code_emb[i](ids_critic[:, i, :])
                            for i in range(self.vq_layers)
                        ],
                        dim=1,
                    ),
                    dim=1,
                )
                scores = token_critic_fn(
                    code_embeds_critic,
                    audio_embeds,
                    task_embeds=task_embeds,
                    self_cond_embed=self_cond_embed,
                    conditioning_token_ids=cond_ids,
                    cond_scale=cond_scale,
                )  # [b, vq_layers, n, 1]
                # remove prompt positions from scores
                scores = scores[:, :, self.prompt_len :, :]
                scores = scores.reshape(batch, self.vq_layers * self.seq_len, -1)
                scores = rearrange(scores, "... 1 -> ...")
            else:
                # Update scores for the next iteration
                probs_without_temperature = logits.softmax(dim=-1)
                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                scores = rearrange(scores, "... 1 -> ...")

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert self.no_mask_token_prob > 0.0, (
                        "Without training with some of the non-masked tokens forced to predict, "
                        "the logits may not be meaningful for these tokens."
                    )

        ids = rearrange(ids, "b (i j) -> b i j", i=self.vq_layers, j=self.seq_len)
        # Decode the final ids to get audios
        with torch.no_grad():
            audios = self.decode(ids)
        return ids, audios

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        noisy_audios: torch.Tensor,
        task_type,  # [batch]
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        timesteps=20,
        cond_scale=1,
        force_not_use_token_critic=False,
        critic_noise_scale=1,
    ):
        # begin with all image token ids masked

        device = next(self.parameters()).device

        batch = noisy_audios.shape[0]

        seq_len_total = self.seq_len + self.prompt_len

        # shape = (batch, self.vq_layers, self.seq_len)

        seq_len = self.vq_layers * seq_len_total
        shape = (batch, seq_len)

        ids = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        cond_ids = None

        # text_embeds = self.transformer.encode_text(texts)

        demask_fn = self.transformer.forward_with_cond_scale

        use_token_critic = self.self_critic and not force_not_use_token_critic
        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        self_cond_embed = None

        task_embeds = (
            self.task_emb(task_type).unsqueeze(1).repeat(1, seq_len_total, 1)
        )  # [b, n, dim]
        audio_embeds, audio_embeds_proj = self.audio_encoder(
            noisy_audios, task_embeds
        )  # list([b, n, dim]), list([b, n, dim])
        # for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):
        for timestep, steps_until_x0 in zip(
            torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))
        ):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            x = ids
            x = x.reshape(batch, self.vq_layers, seq_len_total)

            # encode x & noisy audio
            code_embeds = torch.sum(
                torch.stack(
                    [self.code_emb[i](x[:, i, :]) for i in range(self.vq_layers)], dim=1
                ),
                dim=1,
            )  # [b, n, dim]

            # x = code_embeds + audio_embeds

            logits, embed = demask_fn(
                code_embeds,
                audio_embeds,
                task_embeds=task_embeds,
                self_cond_embed=self_cond_embed,
                conditioning_token_ids=cond_ids,
                cond_scale=cond_scale,
                return_embed=True,
            )

            self_cond_embed = embed if self.self_cond else None

            logits = logits.reshape(batch, seq_len, -1)

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_id

            ids = torch.where(is_mask, pred_ids, ids)

            if use_token_critic:
                ids_critic = ids.reshape(batch, self.vq_layers, seq_len_total)
                code_embeds_critic = torch.sum(
                    torch.stack(
                        [
                            self.code_emb[i](ids_critic[:, i, :])
                            for i in range(self.vq_layers)
                        ],
                        dim=1,
                    ),
                    dim=1,
                )
                scores = token_critic_fn(
                    code_embeds_critic,
                    audio_embeds,
                    task_embeds=task_embeds,
                    self_cond_embed=self_cond_embed,
                    conditioning_token_ids=cond_ids,
                    cond_scale=cond_scale,
                )
                scores = scores.reshape(batch, seq_len, -1)
                scores = rearrange(scores, "... 1 -> ...")
                scores = scores + (
                    uniform(scores.shape, device=device) - 0.5
                ) * critic_noise_scale * (steps_until_x0 / timesteps)
            else:
                probs_without_temperature = logits.softmax(dim=-1)

                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                scores = rearrange(scores, "... 1 -> ...")

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert (
                        self.no_mask_token_prob > 0.0
                    ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        # get ids

        ids = rearrange(ids, "b (i j) -> b i j", i=self.vq_layers, j=seq_len_total)

        with torch.no_grad():
            audios = self.decode(ids)
        return ids, audios

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if "noisy_embed_provider" not in k and "vq_model" not in k
        }
        return filtered_state_dict

    def encode(self, clean_audios: torch.Tensor):
        with torch.no_grad():
            x = self.vq_model.preprocess(clean_audios, 44100)
            z, codes, latents, _, _ = self.vq_model.encode(x)
        return codes

    def decode(self, codes: torch.Tensor):
        with torch.no_grad():
            z_q, _, _ = self.vq_model.quantizer.from_codes(codes)
            audios = self.vq_model.decode(z_q)
        return audios

    def prompt_bandwidth_limitation(
        self, clean_audios: torch.Tensor, rand_prompt: torch.Tensor
    ):
        prompt_len_audio = self.audio_encoder.hop_length * self.prompt_len
        device = clean_audios.device
        if not hasattr(self, "resampler"):
            self.resampler = {
                rate: (
                    torchaudio.transforms.Resample(orig_freq=44100, new_freq=rate).to(
                        device
                    ),
                    torchaudio.transforms.Resample(orig_freq=rate, new_freq=44100).to(
                        device
                    ),
                )
                for rate in self.prompt_bandwidth_limitation_rates
            }

        for i in range(clean_audios.shape[0]):
            if rand_prompt[i] and random() < self.prompt_bandwidth_limitation_prob:
                target_sr = random_choice(self.prompt_bandwidth_limitation_rates)
                prompt_audio = clean_audios[i, :, :prompt_len_audio].to(device)
                resampler, resampler_back = self.resampler[target_sr]
                prompt_audio_resampled = resampler_back(resampler(prompt_audio))
                resampled_len = prompt_audio_resampled.shape[-1]

                if prompt_audio_resampled.shape[-1] < prompt_len_audio:
                    padding = torch.zeros(
                        (clean_audios.shape[1], prompt_len_audio - resampled_len),
                        device=clean_audios.device,
                    )
                    prompt_audio_resampled = torch.cat(
                        [prompt_audio_resampled, padding], dim=-1
                    )

                clean_audios[i, :, :prompt_len_audio] = prompt_audio_resampled[
                    :, :prompt_len_audio
                ]

        return clean_audios

    def forward(
        self,
        clean_audios: torch.Tensor,  # [batch, 1, audio_len]
        noisy_audios: torch.Tensor,
        task_type,  # [batch]
        rand_prompt=None,  # [batch]
        ignore_index=-1,
        cond_drop_prob=None,
    ):
        """
        audio_codes: [batch, audio_len]
        noisy_audios: [batch, audio_len]
        """

        if rand_prompt is None:
            rand_prompt = torch.rand((batch,), device=device) < self.prompt_prob

        if self.prompt_bandwidth_limitation_prob > 0.0:
            clean_audios = self.prompt_bandwidth_limitation(clean_audios, rand_prompt)

        audio_codes = self.encode(clean_audios)

        # get some basic variables
        assert len(audio_codes.shape) == 3
        assert audio_codes.shape[1] == self.vq_layers
        assert audio_codes.shape[2] == self.seq_len + self.prompt_len

        ids = rearrange(audio_codes, "b ... -> b (...)")  # [batch, vq_layers*n]
        batch, seq_len_total = ids.shape  # seq_len_total = seq_len + prompt_len
        device = ids.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        prompt_len = self.prompt_len
        seq_len = self.seq_len

        seq_len_to_mask = torch.where(
            rand_prompt,
            torch.full((batch,), seq_len * self.vq_layers, device=device),
            torch.full((batch,), seq_len_total, device=device),
        )

        # prepare mask
        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (
            (rand_mask_probs * seq_len_to_mask).round().clamp(min=1).long()
        )

        batch_randperm = torch.rand((batch, seq_len_total), device=device).argsort(
            dim=-1
        )
        mask = torch.zeros((batch, seq_len_total), device=device, dtype=torch.bool)

        # Masking for samples with prompts
        if rand_prompt.any():
            idxs_with_prompt = rand_prompt.nonzero(as_tuple=True)[0]
            num_token_masked_with_prompt = num_token_masked[rand_prompt]
            # Adjust positions for target part
            target_positions = batch_randperm[idxs_with_prompt, prompt_len:]
            target_mask = target_positions < rearrange(
                num_token_masked_with_prompt, "b -> b 1"
            )
            # Set mask for target positions
            mask[idxs_with_prompt, prompt_len:] = target_mask

            # Process noisy_audios
            prompt_len_audio = self.audio_encoder.hop_length * self.prompt_len
            # Set the prompt part of noisy_audios to zeros
            noisy_audios[idxs_with_prompt, :, :prompt_len_audio] = 0

        # Masking for samples without prompts
        if (~rand_prompt).any():
            idxs_without_prompt = (~rand_prompt).nonzero(as_tuple=True)[0]
            num_token_masked_without_prompt = num_token_masked[~rand_prompt]
            positions = batch_randperm[idxs_without_prompt]
            sample_mask = positions < rearrange(
                num_token_masked_without_prompt, "b -> b 1"
            )
            # Set mask for all positions
            mask[idxs_without_prompt] = sample_mask

        # Prepare labels
        labels = ids.clone()
        labels[~mask] = ignore_index  # [batch, vq_layers*n]
        # For samples with prompts, set prompt labels to ignore_index
        # labels[rand_prompt, :prompt_len] = ignore_index
        labels = labels.reshape(batch, self.vq_layers, self.seq_len + self.prompt_len)
        labels[rand_prompt, :, :prompt_len] = ignore_index
        labels = labels.reshape(batch, -1)

        # Apply no_mask_token_prob if applicable
        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        # Replace masked positions with mask_id
        mask_id = self.transformer.mask_id
        x = ids.clone()
        x[mask] = mask_id

        # Reshape x back to [batch, vq_layers, seq_len_total]
        x = x.reshape(batch, self.vq_layers, self.seq_len + self.prompt_len)

        # encode x & noisy audio
        code_embeds = torch.sum(
            torch.stack(
                [self.code_emb[i](x[:, i, :]) for i in range(self.vq_layers)], dim=1
            ),
            dim=1,
        )  # [b, n, dim]
        task_embeds = (
            self.task_emb(task_type)
            .unsqueeze(1)
            .repeat(1, self.seq_len + self.prompt_len, 1)
        )  # [b, n, dim]
        audio_embeds, audio_embeds_proj = self.audio_encoder(
            noisy_audios, task_embeds
        )  # list([b, n, dim]), list([b, n, dim])

        # get loss

        ce_loss, logits = self.transformer(
            code_embeds,
            audio_embeds,
            task_embeds=task_embeds,
            self_cond_embed=None,
            conditioning_token_ids=None,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            return_logits=True,
        )

        if self.self_critic:
            logits = logits.reshape(batch, seq_len_total, -1)
            sampled_ids = gumbel_sample(logits, temperature=random())
            x_flatten = x.reshape(batch, -1)

            critic_input = torch.where(mask, sampled_ids, x_flatten)  # [b, n]
            critic_labels = (ids != critic_input).float()
            critic_labels = critic_labels.reshape(
                batch, self.vq_layers, self.seq_len + self.prompt_len
            )
            critic_labels[rand_prompt, :, :prompt_len] = ignore_index

            critic_input = critic_input.reshape(
                batch, self.vq_layers, self.seq_len + self.prompt_len
            )
            code_embeds_critic = torch.sum(
                torch.stack(
                    [
                        self.code_emb[i](critic_input[:, i, :])
                        for i in range(self.vq_layers)
                    ],
                    dim=1,
                ),
                dim=1,
            )

            bce_loss = self.token_critic(
                code_embeds_critic,
                audio_embeds,
                task_embeds=task_embeds,
                self_cond_embed=None,
                conditioning_token_ids=None,
                labels=critic_labels,
                cond_drop_prob=cond_drop_prob,
                ignore_index=ignore_index,
            )

            return (
                ce_loss,
                audio_embeds_proj,
                rand_prompt,
                prompt_len,
                bce_loss,
                self.critic_loss_weight,
            )
        else:
            # Return necessary outputs for loss computation
            return ce_loss, audio_embeds_proj, rand_prompt, prompt_len

    def calculate_critic_score(
        self,
        noisy_audios: torch.Tensor,
        task_type,  # [batch]
        ignore_index=-1,
    ):
        audio_codes = self.encode(noisy_audios)
        assert audio_codes.shape[2] == self.seq_len + self.prompt_len

        task_embeds = (
            self.task_emb(task_type)
            .unsqueeze(1)
            .repeat(1, self.seq_len + self.prompt_len, 1)
        )  # [b, n, dim]
        audio_embeds, audio_embeds_proj = self.audio_encoder(noisy_audios, task_embeds)

        critic_input = audio_codes.reshape(
            audio_codes.shape[0], self.vq_layers, self.seq_len + self.prompt_len
        )
        code_embeds_critics = torch.sum(
            torch.stack(
                [
                    self.code_emb[i](critic_input[:, i, :])
                    for i in range(self.vq_layers)
                ],
                dim=1,
            ),
            dim=1,
        )

        pseudo_labels = torch.zeros_like(audio_codes, device=audio_codes.device).float()

        bce_loss, logits = self.token_critic(
            code_embeds_critics,
            audio_embeds,
            task_embeds=task_embeds,
            ignore_index=ignore_index,
            self_cond_embed=None,
            conditioning_token_ids=None,
            labels=pseudo_labels,
            return_logits=True,
        )

        return bce_loss, logits
