# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import shutil
import torch
import time
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from dualcodec import BaseTrainer
import safetensors
import numpy as np
from einops import rearrange
from .flatten_patterns import offset_codes


def extract_codes_official_dac(model, audio):
    audio = rearrange(audio, "b t -> b 1 t")
    compressed = model.encode(audio)[1]  # (b, n_q, t)
    return compressed  # (b, n_q, t)


def extract_codes_modular_dac(model, audio):
    audio = rearrange(audio, "b t -> b 1 t")
    semantic_codes, acoustic_codes = model.encode(audio)  # (b, q, t) both
    return semantic_codes, acoustic_codes  # (b, q, t)


class Trainer(BaseTrainer):
    """Trainer"""

    def __init__(self, args=None, cfg=None, **kwargs):
        """
            Initializes the model with the given arguments and configuration.

        Args:
            args (argparse.Namespace, optional): Arguments to be passed on to the model. Defaults to None.
            cfg (dict, optional): Configuration dictionary containing parameters for the model. Defaults to None.
        """
        super().__init__(args, cfg)
        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            print("skip semantic normalize")
        for key in self.cfg.semantic_model:
            if isinstance(self.cfg.semantic_model[key], torch.nn.Module) or isinstance(
                self.cfg.semantic_model[key], torch.Tensor
            ):
                self.cfg.semantic_model[key] = self.cfg.semantic_model[key].to(
                    self.accelerator.device
                )

    def _accelerator_prepare(self):
        """
        Returns: None
        """
        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
        )

    def _build_scheduler(self):
        """
        Returns: None
        """
        return None

    # def _build_criterion(self):
    #     pass  # loss is directly returned from model

    def _build_model(self):
        """
        Returns: None
        """
        return self.cfg.model  # llama model

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_codes_dac(self, input_features, attention_mask, audio):
        vq_emb = self.cfg.semantic_model["model"](
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.cfg.semantic_model["output_idx"]]  # (B, T, C)
        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            pass
        else:
            feat = (feat - self.cfg.semantic_model["mean"]) / self.cfg.semantic_model[
                "std"
            ]
        feat = feat.transpose(1, 2)
        feat = torch.nn.functional.avg_pool1d(
            feat,
            self.cfg.semantic_model["repcodec_model"].semantic_downsample_factor,
            self.cfg.semantic_model["repcodec_model"].semantic_downsample_factor,
        )

        audio = rearrange(audio, "b t -> b 1 t")
        semantic_codes, acoustic_codes = self.cfg.semantic_model[
            "repcodec_model"
        ].encode(audio, semantic_repr=feat)
        semantic_codes = rearrange(semantic_codes, "b 1 t -> b t")
        acoustic_codes = rearrange(acoustic_codes, "b q t -> b t q")
        return semantic_codes, acoustic_codes

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_semantic_code(self, input_features, attention_mask):
        """
        Extract semantic code from input features.
        This function is marked with @torch.no_grad() as it doesn't require gradients.

        Args:
            input_features (torch.Tensor, shape=(B, T, C)): Input features, where B is batch size, T is time dimension, C is channel dimension.
            attention_mask (torch.Tensor, shape=(B, T)): Attention mask, where 0 indicates invalid features and non-zero indicates valid features.

        Returns:
            tuple (torch.Tensor, shape=(B, T)): Returns a tuple containing semantic code and optional quantization indices.
                - semantic_code (torch.Tensor, shape=(B, T)): Semantic code, where B is batch size, T is time dimension.
                - rep_index (Optional, torch.Tensor, shape=(B, T)): For each time step, returns quantization indices if they exist; otherwise returns None.
        """
        vq_emb = self.cfg.semantic_model["model"](
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.cfg.semantic_model["output_idx"]]  # (B, T, C)

        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            pass
        else:
            feat = (feat - self.cfg.semantic_model["mean"]) / self.cfg.semantic_model[
                "std"
            ]

        if hasattr(self.cfg, "use_our_codec"):
            feat = torch.nn.functional.avg_pool1d(
                feat.transpose(1, 2),
                self.cfg.semantic_model["repcodec_model"].semantic_downsample_factor,
                self.cfg.semantic_model["repcodec_model"].semantic_downsample_factor,
            )

            semantic_code = self.cfg.semantic_model["repcodec_model"].semantic_quantize(
                feat
            )
        else:
            semantic_code, _ = self.cfg.semantic_model["repcodec_model"].quantize(
                feat
            )  # (B, T)
        return semantic_code, None

    def _train_step(self, batch):
        """Returns: dict('speech', 'speech_len', 'phone_ids', 'phone_lens')
        speech: [B, T]
        speech_len: [B]
        phone_ids: [B, T]
        phone_lens: [B]
        """
        device = self.accelerator.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        input_features = batch["input_features"]
        attention_mask = batch["attention_mask"]
        if hasattr(self.cfg, "use_our_codec") and self.cfg.use_our_codec:
            batch["speech_token_len"] = (
                batch["speech_token_len"]
                // self.cfg.semantic_model["repcodec_model"].semantic_downsample_factor
            )
        if hasattr(self.cfg, "use_modular_dac") and self.cfg.use_modular_dac:
            assert not hasattr(self.cfg, "use_our_codec") or not self.cfg.use_our_codec
            assert (
                not hasattr(self.cfg, "use_official_dac")
                or not self.cfg.use_official_dac
            )
            semantic_codes, acoustic_codes = extract_codes_modular_dac(
                self.cfg.semantic_model["repcodec_model"], batch["speech"]
            )
            semantic_code = semantic_codes[:, 0]  # b,t
        if hasattr(self.cfg, "use_official_dac"):
            assert not hasattr(self.cfg, "use_our_codec") or not self.cfg.use_our_codec
            acoustic_codes = extract_codes_official_dac(
                self.cfg.semantic_model["repcodec_model"], batch["speech"]
            )
            semantic_code = acoustic_codes[:, 0]  # b,t
        if hasattr(self.cfg, "use_acoustic_codes") and self.cfg.use_acoustic_codes:
            semantic_codes = rearrange(semantic_codes, "b t -> b t 1")
            # Concatenate semantic and acoustic codes along the codec layer dimension
            num_codec_layers = len(self.cfg.offset_sizes)
            semantic_code = torch.cat([semantic_codes, acoustic_codes], dim=-1)[
                ..., :num_codec_layers
            ]

            # # Apply layer-specific offsets before rearranging
            # offsetted_code = []
            # for i in range(num_codec_layers):
            #     # Get the offset size for the current layer
            #     layer_offset = int(np.sum(self.cfg.offset_sizes[:i]))

            #     # Extract the current layer's codes
            #     current_layer_code = semantic_code[..., i]  # Shape (batch_size, T)

            #     # Apply the offset
            #     current_layer_code += layer_offset

            #     # Append the offsetted layer code
            #     offsetted_code.append(current_layer_code)

            # # Concatenate all the offsetted layers along the codec layer dimension
            # offsetted_code = torch.stack(offsetted_code, dim=-1)

            semantic_code = offset_codes(semantic_code, self.cfg.offset_sizes)

            # Rearrange (flatten the codec layers to the time dimension)
            semantic_code = rearrange(semantic_code, "b t q -> b (t q)")

            batch["speech_token_len"] *= num_codec_layers

            del batch["speech"]
        else:
            semantic_code, _ = self._extract_semantic_code(
                input_features, attention_mask
            )  # if len(semantic_code) == 2: [B, T]; else 3: [N, B, T]

        del batch["input_features"]
        del batch["attention_mask"]
        batch["speech_token"] = semantic_code
        out = self.model(batch, device=device)

        return out["loss"], {"Train/Batch Size": input_features.shape[0]} | out["acc"]

    def _test_step(self, batch):
        raise NotImplementedError

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        epoch_sum_loss = 0.0
        return epoch_sum_loss

    def _inference(self):
        pass

    def test_loop(self):
        return
        self.model.eval()
        for batch in self.train_dataloader:
            self._test_step(batch)
