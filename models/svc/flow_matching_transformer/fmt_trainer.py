# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import torchvision
import random
import numpy as np
import accelerate

import whisper
from models.base.base_trainer import BaseTrainer
from models.codec.coco.coco_dataset import CocoDataset, CocoCollator
from models.svc.flow_matching_transformer.fmt_model import FlowMatchingTransformer
from models.codec.melvqgan.melspec import MelSpectrogram
from models.codec.coco.rep_coco_model import CocoContentStyle, CocoContent


class FlowMatchingTransformerTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(FlowMatchingTransformerTrainer, self).__init__(args, cfg)

        self._build_input_model()
        self._build_output_model()

        self.use_whisper_perturb = getattr(
            self.cfg.model.flow_matching_transformer, "whisper_perturb", True
        )
        if self.accelerator.is_main_process:
            self.logger.info(f"Use whisper perturb: {self.use_whisper_perturb}")

    def _build_model(self):
        model = FlowMatchingTransformer(cfg=self.cfg.model.flow_matching_transformer)

        if "pretrained_path" in self.cfg.model.flow_matching_transformer:
            accelerate.load_checkpoint_and_dispatch(
                model, self.cfg.model.flow_matching_transformer.pretrained_path
            )
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Load pretrained model from {self.cfg.model.flow_matching_transformer.pretrained_path}"
                )

        return model

    def _build_dataset(self):
        return CocoDataset, CocoCollator

    def spec_augment(self, mel, height):
        """
        Args:
            mel: tensor (..., n_mels, frames)
            height: int 68-92 for default 80 mels
        """
        tgt = torchvision.transforms.functional.resize(mel, (height, mel.shape[-1]))
        if height >= mel.shape[-2]:
            return tgt[:, : mel.shape[-2], :]
        else:
            silence = tgt[:, -1:, :].repeat(1, mel.shape[-2] - height, 1)
            silence += torch.randn_like(silence) / 10
            return torch.cat((tgt, silence), 1)

    @torch.no_grad()
    @torch.amp.autocast("cuda", dtype=torch.bfloat16)  # Specify bf16
    def _extract_whisper_features(self, wavs, frame_lens, spec_perturb=True):
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

        if spec_perturb:
            height = random.randint(68, 92)
            batch_mel = self.spec_augment(batch_mel, height)

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

    def _build_output_model(self):
        self.mel_model = MelSpectrogram(
            sampling_rate=self.cfg.preprocess.sample_rate,
            n_fft=self.cfg.preprocess.n_fft,
            num_mels=self.cfg.preprocess.num_mels,
            hop_size=self.cfg.preprocess.hop_size,
            win_size=self.cfg.preprocess.win_size,
            fmin=self.cfg.preprocess.fmin,
            fmax=self.cfg.preprocess.fmax,
        )
        self.mel_model.eval()
        self.mel_model.to(self.accelerator.device)

    @torch.no_grad()
    def _extract_mel_feature(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.cfg.preprocess.mel_mean) / math.sqrt(
            self.cfg.preprocess.mel_var
        )
        return mel_feature

    def _build_input_model(self):
        ## Whisper ##
        self.whisper_model = whisper.load_model(
            "medium", self.accelerator.device
        )  # 1024 dim
        self.whisper_model.eval()

        if self.accelerator.mixed_precision == "bf16":
            self.whisper_model = self.whisper_model.to(torch.bfloat16)

        self.use_normed_whisper = getattr(
            self.cfg.model.coco, "use_normed_whisper", False
        )
        if self.use_normed_whisper:
            whisper_stats = torch.load(
                self.cfg.model.coco.whisper_stats_path,
                map_location=self.accelerator.device,
            )
            self.whisper_mean = whisper_stats["mean"]  # (1024,)
            self.whisper_std = whisper_stats["std"]  # (1024,)

            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Use normed whisper, mean: {self.whisper_mean.shape}, std: {self.whisper_std.shape}"
                )

        ## Coco ##
        self.coco_model_type = getattr(
            self.cfg.model.coco, "coco_type", "content_style"
        )
        if self.coco_model_type == "content_style":
            model = CocoContentStyle(
                cfg=self.cfg.model.coco, construct_only_for_quantizer=True
            )
        elif self.coco_model_type == "content":
            model = CocoContent(
                cfg=self.cfg.model.coco, construct_only_for_quantizer=True
            )
        else:
            raise ValueError(f"Unknown coco type: {self.coco_model_type}")

        self.coco_model = model
        self.coco_model.eval()
        self.coco_model.to(self.accelerator.device)
        accelerate.load_checkpoint_and_dispatch(
            self.coco_model, self.cfg.model.coco.pretrained_path
        )

    @torch.no_grad()
    def _extract_coco_codec(self, wav16k, chromagram_feats, target_frame_lens):
        """
        Args:
            wav16k: [B, T]
            chromagram_feats: [B, T, 24]
            target_frame_lens: [B]
        Returns:
            codecs: [B, T]. Note that codecs might be not at 50Hz!
        """
        whisper_feats = self._extract_whisper_features(
            wav16k, target_frame_lens, spec_perturb=self.use_whisper_perturb
        )  # [B, T, D]

        if self.coco_model_type == "content_style":
            codecs, _ = self.coco_model.quantize(
                whisper_feats.to(torch.float32), chromagram_feats.to(torch.float32)
            )
        elif self.coco_model_type == "content":
            codecs, _ = self.coco_model.quantize(whisper_feats.to(torch.float32))
        else:
            raise ValueError(f"Unknown coco type: {self.coco_model_type}")

        return codecs

    def _build_semantic_model(self):
        from transformers import Wav2Vec2BertModel

        self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model.eval()
        self.semantic_model.to(self.accelerator.device)

        if self.accelerator.mixed_precision == "bf16":
            self.semantic_model = self.semantic_model.to(torch.bfloat16)

        stat_mean_var = torch.load("models/tts/maskgct/ckpt/wav2vec2bert_stats.pt")
        self.semantic_mean = stat_mean_var["mean"]
        self.semantic_std = torch.sqrt(stat_mean_var["var"])
        self.semantic_mean = self.semantic_mean.to(self.accelerator.device)
        self.semantic_std = self.semantic_std.to(self.accelerator.device)

        if self.accelerator.is_main_process:
            self.logger.info(
                f"Use semantic model, mean: {self.semantic_mean.shape}, std: {self.semantic_std.shape}"
            )

    @torch.no_grad()
    @torch.amp.autocast("cuda", dtype=torch.bfloat16)  # Specify bf16
    def _extract_semantic_feature(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        return feat

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        speech = batch["wav"]  # [B, T]
        x_mask = batch["mask"]  # [B, n_frames]
        frame_lens = torch.sum(x_mask, dim=1).int()  # [B]
        mel_feat = self._extract_mel_feature(speech)

        chromagram_feats = batch["chromagram"]  # [B, T, 24]
        cond_code = self._extract_coco_codec(
            batch["wav_16000"], chromagram_feats, frame_lens
        )

        # torch.cuda.empty_cache()

        fm_output_dict = self.model(
            x=mel_feat,
            x_mask=x_mask,
            cond_code=cond_code,
        )
        noise, x, flow_pred, final_mask, prompt_len = fm_output_dict["output"]
        final_mask = final_mask.squeeze(-1)

        flow_gt = x - (1 - self.cfg.model.flow_matching_transformer.sigma) * noise

        # [B, n_frames, D]
        diff_loss = F.l1_loss(
            flow_pred, flow_gt, reduction="none"
        ).float() * final_mask.unsqueeze(-1)
        diff_loss = torch.mean(diff_loss, dim=2).sum() / final_mask.sum()

        total_loss += diff_loss
        train_losses["diff_loss"] = diff_loss

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 0.2
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()

        train_losses["batch_size"] = mel_feat.shape[0]
        train_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return (total_loss.item(), train_losses, train_stats)
