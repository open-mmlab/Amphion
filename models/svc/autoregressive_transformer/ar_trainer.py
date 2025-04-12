# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import whisper

from models.base.base_trainer import BaseTrainer
from models.codec.coco.coco_dataset import CocoDataset, CocoCollator
from models.svc.autoregressive_transformer.ar_model import AutoregressiveTransformer
from models.codec.coco.rep_coco_model import CocoContentStyle, CocoContent, CocoStyle

import accelerate


class AutoregressiveTransformerTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(AutoregressiveTransformerTrainer, self).__init__(args, cfg)

        self._build_whisper_model()
        self._build_coco_model()

    def _build_model(self):
        model = AutoregressiveTransformer(cfg=self.cfg.model.autoregressive_transformer)
        return model

    def _build_dataset(self):
        return CocoDataset, CocoCollator

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)  # Specify bf16
    def _extract_whisper_features(self, wavs, frame_lens):
        """
        Args:
            wavs: (B, T) at 16khz. Note that the max duration should be 30s
            frame_lens: (B,)
        Returns:
            features: (B, T, D)
        """
        # wavs: (batch, max_len)
        wavs = whisper.pad_or_trim(wavs)
        # batch_mel: (batch, 80, 3000)
        batch_mel = whisper.log_mel_spectrogram(wavs, device=self.accelerator.device)

        with torch.no_grad():
            # (batch, 1500, 1024)
            features = self.whisper_model.embed_audio(batch_mel)

        max_len = int(frame_lens.max().item())
        mask = torch.arange(features.size(1), device=features.device).expand(
            len(frame_lens), -1
        ) < frame_lens.unsqueeze(1)
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))

        if features.shape[1] >= max_len:
            features = features[:, :max_len, :]
        else:
            padding_frames = max_len - features.shape[1]
            last_frame = features[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            features = torch.cat([features, padding], dim=1)

        if self.use_normed_whisper:
            features = (features - self.whisper_mean) / self.whisper_std

        return features

    def _build_whisper_model(self):
        ## Whisper ##
        self.whisper_model = whisper.load_model(
            "medium", self.accelerator.device
        )  # 1024 dim
        self.whisper_model.eval()

        if self.accelerator.mixed_precision == "bf16":
            self.whisper_model = self.whisper_model.to(torch.bfloat16)

        self.use_normed_whisper = self.cfg.model.coco_content_style.use_normed_whisper
        if self.use_normed_whisper:
            whisper_stats = torch.load(
                self.cfg.model.coco_content_style.whisper_stats_path,
                map_location=self.accelerator.device,
            )
            self.whisper_mean = whisper_stats["mean"]  # (1024,)
            self.whisper_std = whisper_stats["std"]  # (1024,)

            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Use normed whisper, mean: {self.whisper_mean.shape}, std: {self.whisper_std.shape}"
                )

    def _build_coco_model(self):
        ## Content-Style Tokenizer ##
        self.coco_content_style = CocoContentStyle(
            cfg=self.cfg.model.coco_content_style, construct_only_for_quantizer=True
        )
        self.coco_content_style.eval()
        self.coco_content_style.to(self.accelerator.device)
        accelerate.load_checkpoint_and_dispatch(
            self.coco_content_style, self.cfg.model.coco_content_style.pretrained_path
        )

        ## Content Tokenizer ##
        if self.cfg.model.use_content_tokens_as_input:
            self.content_tokenizer = CocoContent(
                cfg=self.cfg.model.coco_content, construct_only_for_quantizer=True
            )
            self.content_tokenizer.eval()
            self.content_tokenizer.to(self.accelerator.device)
            accelerate.load_checkpoint_and_dispatch(
                self.content_tokenizer, self.cfg.model.coco_content.pretrained_path
            )

        ## Style Tokenizer ##
        if self.cfg.model.use_style_tokens_as_input:
            self.style_tokenizer = CocoStyle(
                cfg=self.cfg.model.coco_style, construct_only_for_quantizer=True
            )
            self.style_tokenizer.eval()
            self.style_tokenizer.to(self.accelerator.device)
            accelerate.load_checkpoint_and_dispatch(
                self.style_tokenizer, self.cfg.model.coco_style.pretrained_path
            )

    @torch.no_grad()
    def _extract_coco_codec(
        self, coco_cfg, coco_model, whisper_feats, chromagram_feats, frame_lens_50hz
    ):
        """
        Args:
            coco_model_type: "content_style" or "content" or "style"
            whisper_feats: [B, T, D]
            chromagram_feats: [B, T, 24]
            frame_lens_50hz: [B]
        Returns:
            codecs: [B, T]. Note that codecs might be not at 50Hz!
            codec_masks: [B, T]
        """
        coco_model_type = coco_cfg.coco_type
        if coco_model_type == "content_style":
            codecs, _ = coco_model.quantize(
                whisper_feats.to(torch.float32), chromagram_feats.to(torch.float32)
            )
        elif coco_model_type == "content":
            codecs, _ = coco_model.quantize(whisper_feats.to(torch.float32))
        elif coco_model_type == "style":
            codecs, _ = coco_model.quantize(chromagram_feats.to(torch.float32))
        else:
            raise ValueError(f"Unknown coco type: {coco_model_type}")

        coco_downsample_rate = coco_cfg.downsample_rate
        codecs_frame_lens = frame_lens_50hz // coco_downsample_rate + 1

        T = codecs.shape[1]
        arange_tensor = torch.arange(T).expand(codecs.shape[0], T).to(codecs)
        codec_masks = (
            arange_tensor < codecs_frame_lens.unsqueeze(-1)
        ).int()  # 1 means valid

        return codecs, codec_masks

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        speech = batch["wav"]  # [B, T]
        x_mask = batch["mask"]  # [B, n_frames]
        frame_lens = torch.sum(x_mask, dim=1).int()  # [B]

        chromagram_feats = batch["chromagram"]  # [B, T, 24]
        whisper_feats = self._extract_whisper_features(
            batch["wav_16000"], frame_lens
        )  # [B, T, D]

        # [B, T]
        content_style_ids, content_style_masks = self._extract_coco_codec(
            coco_cfg=self.cfg.model.coco_content_style,
            coco_model=self.coco_content_style,
            whisper_feats=whisper_feats,
            chromagram_feats=chromagram_feats,
            frame_lens_50hz=frame_lens,
        )

        if getattr(self.cfg.model, "train_both_conversion_and_synthesis", False):
            raise NotImplementedError(
                "train_both_conversion_and_synthesis is not implemented"
            )

        elif getattr(self.cfg.model, "train_only_conversion", False):
            if self.cfg.model.use_style_tokens_as_input and random.random() < 0.5:
                style_ids, style_masks = self._extract_coco_codec(
                    coco_cfg=self.cfg.model.coco_style,
                    coco_model=self.style_tokenizer,
                    whisper_feats=whisper_feats,
                    chromagram_feats=chromagram_feats,
                    frame_lens_50hz=frame_lens,
                )
                task = "SVC"
            else:
                style_ids = None
                style_masks = None
                task = "VC"

            content_ids, content_masks = self._extract_coco_codec(
                coco_cfg=self.cfg.model.coco_content,
                coco_model=self.content_tokenizer,
                whisper_feats=whisper_feats,
                chromagram_feats=chromagram_feats,
                frame_lens_50hz=frame_lens,
            )

        elif getattr(self.cfg.model, "train_only_synthesis", False):
            if self.cfg.model.use_style_tokens_as_input and random.random() < 0.5:
                style_ids, style_masks = self._extract_coco_codec(
                    coco_cfg=self.cfg.model.coco_style,
                    coco_model=self.style_tokenizer,
                    whisper_feats=whisper_feats,
                    chromagram_feats=chromagram_feats,
                    frame_lens_50hz=frame_lens,
                )
                task = "SVS"
            else:
                style_ids = None
                style_masks = None
                task = "TTS"

            content_ids = batch["phone_id"]
            content_masks = batch["phone_mask"]

        else:
            raise ValueError("Unknown task")

        # torch.cuda.empty_cache()

        out = self.model(
            content_ids,
            content_masks,
            style_ids,
            style_masks,
            content_style_ids,
            content_style_masks,
        )

        total_loss += out.loss
        train_losses["ce_loss"] = total_loss
        train_losses["ce_loss_{}".format(task)] = total_loss

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()

        train_losses["batch_size"] = speech.shape[0]
        train_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return (total_loss.item(), train_losses, train_stats)
