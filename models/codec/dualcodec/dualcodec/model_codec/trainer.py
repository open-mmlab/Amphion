################################################################################
#
# Copyright (c) 2024 Amphion. All Rights Reserved
#
################################################################################

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
from .discriminator import Discriminator
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict

USE_HINGE_LOSS = False
from audiotools import AudioSignal


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
        torch.backends.cudnn.benchmark = True

        from .loss import GANLoss, MelSpectrogramLoss, MultibandMelSpectrogramLoss

        self.gan_loss = GANLoss(self.cfg.discriminator_model)
        self.spec_loss = MelSpectrogramLoss(
            pow=2,
            mag_weight=1,
            log_weight=1,
            n_mels=[40, 80, 160, 320],
            window_lengths=[256, 512, 1024, 2048],
        )
        self.semantic_spec_loss = MultibandMelSpectrogramLoss(
            # bands=[(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
            # band_weights=[16,8,4,2,1],
            bands=[(0.0, 0.1)],
            band_weights=[1.0],
            loss_fn=nn.MSELoss(),
            pow=2,
            mag_weight=1,
            log_weight=1,
            n_mels=[80, 160, 320],
            window_lengths=[512, 1024, 2048],
        )

        if hasattr(self.cfg, "semantic_model"):
            for key in self.cfg.semantic_model:
                if isinstance(
                    self.cfg.semantic_model[key], torch.nn.Module
                ) or isinstance(self.cfg.semantic_model[key], torch.Tensor):
                    self.cfg.semantic_model[key] = self.cfg.semantic_model[key].to(
                        self.accelerator.device
                    )
        self.distill = False

        self.model_module = self.model
        if hasattr(self.model, "module"):
            self.model_module = self.model.module

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_semantic_code(self, input_features, attention_mask):
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
        return feat

    def _build_model(self):
        """
        Returns: None
        """
        return edict(
            {
                "generator": self.cfg.model,
                "discriminator": self.cfg.discriminator_model,
            }
        )

    def _build_optimizer(self):
        r"""Build optimizer for model."""
        return edict(
            {
                "optimizer_g": self.cfg.train.optimizer(
                    params=self.model.generator.parameters()
                ),
                "optimizer_d": self.cfg.train.optimizer(
                    params=self.model.discriminator.parameters()
                ),
            }
        )

    def _accelerator_prepare(self):
        """
        Returns: None
        """
        (
            self.model,
            self.discriminator,
            self.optimizer,
            self.optimizer_d,
        ) = self.accelerator.prepare(
            self.model.generator,
            self.model.discriminator,
            self.optimizer.optimizer_g,
            self.optimizer.optimizer_d,
        )

    def _build_scheduler(self):
        """
        Returns: None
        """
        return None

    def _train_step(self, batch):
        """
        Args:
        - batch: dict containing the batch data
        -- batch["speech"]: torch.Tensor of shape (B, T)
        -- batch["speech_lens"]: torch.Tensor of shape (B,), contains the length of the unpadded speech
        -- batch["input_features"]: torch.Tensor of shape (B, T, C), extracted by w2v-bert feat extractor
        -- batch["attention_mask"]: torch.Tensor of shape (B, T), attention mask for the input_features, extracted by w2v-bert feat extractor
        """
        optim_g, optim_d = self.optimizer, self.optimizer_d

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.accelerator.device)

        x_wav, audio_lengths = batch["speech"], batch["speech_lens"]

        x_wav = x_wav.float()[:, None, :]

        if self.cfg.semantic_vq:
            input_features = batch["input_features"]
            attention_mask = batch["attention_mask"]
            feat = self._extract_semantic_code(
                input_features, attention_mask
            ).transpose(1, 2)
            feat = torch.nn.functional.avg_pool1d(
                feat,
                self.model_module.semantic_downsample_factor,
                self.model_module.semantic_downsample_factor,
            )
            out_dict, semantic_edict = self.model(
                x_wav,
                semantic_repr=feat,
                bypass_quantize_rate=0.125,
                possibly_no_quantizer=False,  # internal dropout
            )
        else:
            out_dict = self.model(
                x_wav,
            )
            semantic_edict = None

        generator_out = out_dict.x
        commitment_loss = out_dict.penalty
        metrics = out_dict.metrics
        codebook_loss = out_dict["vq/codebook_loss"]
        first_layer_quantized = out_dict["first_layer_quantized"]

        # --------- Discriminator training ------------
        if USE_HINGE_LOSS:
            disc_loss = self.gan_loss.discriminator_hinge_loss(generator_out, x_wav)
        else:
            disc_loss = self.gan_loss.discriminator_loss(generator_out, x_wav)
        self.optimizer_d.zero_grad()
        self.accelerator.backward(disc_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer_d.step()
        self.optimizer_d.zero_grad()

        if USE_HINGE_LOSS:
            adv_g_loss, feat_loss = self.gan_loss.generator_hinge_loss(
                generator_out, x_wav
            )
        else:
            adv_g_loss, feat_loss = self.gan_loss.generator_loss(generator_out, x_wav)
        spec_loss = self.spec_loss(
            AudioSignal(x_wav, 24000), AudioSignal(generator_out, 24000)
        )
        # spec_loss = reconstruction_loss(x_wav, generator_out, args)
        total_loss = (
            0.25 * commitment_loss
            + 1.0 * adv_g_loss
            + 2.0 * feat_loss
            + 15.0 * spec_loss
            + 1.0 * codebook_loss
        )
        # ---------- Generator training ----------------
        if semantic_edict:
            distill_loss = F.mse_loss(feat, semantic_edict["x"])
            total_loss += (
                distill_loss * self.cfg.lambda_distill_loss
                + self.cfg.lambda_semantic_commitment_loss * semantic_edict["penalty"]
                + self.cfg.lambda_semantic_codebook_loss
                + semantic_edict["vq/codebook_loss"]
            )
            metrics.update(
                {
                    "semantic/semantic_commitment_loss": semantic_edict["penalty"],
                    "semantic/semantic_codebook_loss": semantic_edict[
                        "vq/codebook_loss"
                    ],
                    "semantic/semantic_distill_loss": distill_loss,
                }
            )
            if semantic_edict["bypassed_quantize"]:
                metrics.update(
                    {
                        "semantic/spec_loss": spec_loss,
                    }
                )
            if self.cfg.add_semantic_spec_loss and semantic_edict["bypassed_quantize"]:
                semantic_spec_loss = 15.0 * self.semantic_spec_loss(
                    AudioSignal(x_wav, 24000), AudioSignal(generator_out, 24000)
                )
                total_loss += semantic_spec_loss
                metrics.update(
                    {
                        "semantic/semantic_spec_loss": semantic_spec_loss,
                    }
                )

        if self.distill:
            input_features = batch["input_features"]
            attention_mask = batch["attention_mask"]
            feat = self._extract_semantic_code(
                input_features, attention_mask
            ).transpose(1, 2)
            feat = torch.nn.functional.avg_pool1d(
                feat, self.semantic_downsample_factor, self.semantic_downsample_factor
            )
            distill_loss = F.mse_loss(
                feat, first_layer_quantized[..., : feat.shape[-1]]
            )
            total_loss += distill_loss * self.cfg.lambda_distill_loss
        else:
            distill_loss = 0.0

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # print(commitment_loss, spec_loss)
        # print(x_wav.shape[0])
        # breakpoint()
        # print(out_dict.codes)
        # if self.step >= 100:
        #     breakpoint()
        metrics.update(
            {
                "commitment_loss": commitment_loss,
                "spec_loss": spec_loss,
                "feat_loss": feat_loss,
                "adv_g_loss": adv_g_loss,
                "total_loss": total_loss,
                "Train/Batch Size": x_wav.shape[0],
                "disc_loss": disc_loss.item(),
                "distill_loss": distill_loss,
            }
        )
        # print(metrics)

        return None, metrics

    def _load_model(
        self,
        checkpoint_dir: str = None,
        checkpoint_path: str = None,
        resume_type: str = "",
    ):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            try:
                all_ckpts = os.listdir(checkpoint_dir)
                all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
                ls = list(all_ckpts)
                ls = [os.path.join(checkpoint_dir, i) for i in ls]
                ls.sort(
                    key=lambda x: int(x.split("_")[-2].split("-")[-1]), reverse=True
                )
                checkpoint_path = ls[0]
                self.logger.info("Resume from {}".format(checkpoint_path))
            except Exception as e:
                print(
                    "Failed to load checkpoint from {}, starting FROM SCRATCH...".format(
                        checkpoint_dir
                    )
                )
                return None

        if resume_type in ["resume", ""]:
            # Load all the things, including model weights, optimizer, scheduler, and random states.
            try:
                self.accelerator.load_state(input_dir=checkpoint_path)
            except Exception as e:
                print(e)
            # set epoch and step
            from pathlib import Path

            self.epoch = int(Path(checkpoint_path).name.split("_")[0].split("-")[-1])
            if hasattr(self.args, "reset_steps") and self.args.reset_steps:
                self.step = 0
            else:
                self.step = (
                    int(Path(checkpoint_path).name.split("_")[1].split("-")[-1]) + 1
                )

        elif resume_type == "finetune":
            # Load only the model weights
            import safetensors.torch

            safetensors.torch.load_model(
                self.accelerator.unwrap_model(self.model),
                os.path.join(
                    checkpoint_path, self.args.model_1_name
                ),  # location of "model_1.safetensors"
            )
            safetensors.torch.load_model(
                self.accelerator.unwrap_model(self.discriminator),
                os.path.join(
                    checkpoint_path, self.args.model_2_name
                ),  # location of "model_2.safetensors"
            )
            self.logger.info("Loaded model weights for finetune.")

        else:
            raise ValueError("Resume_type must be `resume` or `finetune`.")

        return checkpoint_path

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
