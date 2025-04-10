# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
import math
import torch.nn.functional as F
import torchaudio
from models.base.base_trainer import BaseTrainer
from models.vc.base.vc_emilia_dataset import VCEmiliaDataset, VCCollator
from models.codec.melvqgan.melspec import MelSpectrogram
from models.vc.flow_matching_transformer.fmt_model import FlowMatchingTransformer
from models.codec.kmeans.repcodec_model import RepCodec

import safetensors


class FlowMatchingTransformerTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(FlowMatchingTransformerTrainer, self).__init__(args, cfg)

        # setup input model (such as content-style tokens)
        self._build_input_model()

        # setup output model (such as mels)
        self._build_output_model()

    def _build_model(self):
        model = FlowMatchingTransformer(cfg=self.cfg.model.flow_matching_transformer)
        if (
            hasattr(self.cfg.model.flow_matching_transformer, "pretrained_path")
            and hasattr(
                self.cfg.model.flow_matching_transformer, "use_pretrained_model"
            )
            and self.cfg.model.flow_matching_transformer.use_pretrained_model
        ):
            pretrained_path = self.cfg.model.flow_matching_transformer.pretrained_path
            if ".bin" in pretrained_path:
                model.load_state_dict(torch.load(pretrained_path), strict=False)
            elif ".safetensors" in pretrained_path:
                safetensors.torch.load_model(model, pretrained_path, strict=False)
        return model

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

    @torch.no_grad()
    def _extract_hubert_feature(self, wavs, wav_lens=None, output_layer=18):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            feats: [B, T, D]
        """
        with torch.no_grad():
            feats, _ = self.hubert.extract_features(
                wavs, lengths=wav_lens, num_layers=output_layer
            )
            feats = feats[-1].squeeze()
        return feats

    @torch.no_grad()
    def _extract_hubert_codec(self, wavs, wav_lens=None, output_layer=18):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            codecs: [B, T]
        """
        # Extract features and normalize
        feats = self._extract_hubert_feature(wavs, wav_lens, output_layer)
        feats = (feats - self.feat_norm_mean.to(feats)) / self.feat_norm_std.to(feats)

        # VQ-VAE Tokenizer
        codecs, _ = self.vqvae.quantize(feats)  # (B, T)
        return codecs

    def _build_input_model(self):
        if self.cfg.model.cond_type in ["hubert_features", "hubert_codec"]:
            bundle = torchaudio.pipelines.HUBERT_LARGE
            self.hubert = bundle.get_model()
            self.hubert.eval()
            self.hubert.to(self.accelerator.device)

        if self.cfg.model.cond_type == "hubert_codec":
            # Features normalization
            stat = np.load(self.cfg.model.representation_stat_mean_var_path)
            self.feat_norm_mean = torch.tensor(stat["mean"]).to(self.accelerator.device)
            self.feat_norm_std = torch.tensor(stat["std"]).to(self.accelerator.device)

            # VQ-VAE Tokenizer
            self.vqvae = RepCodec(cfg=self.cfg.model.repcodec)
            self.vqvae.eval()

            pretrained_path = self.cfg.model.repcodec.pretrained_path
            if ".bin" in pretrained_path:
                self.vqvae.load_state_dict(torch.load(pretrained_path))
            elif ".safetensors" in pretrained_path:
                safetensors.torch.load_model(self.vqvae, pretrained_path)
            self.vqvae.to(self.accelerator.device)

    def _build_dataset(self):
        assert self.cfg.train.use_emilia_dataset
        return VCEmiliaDataset, VCCollator

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        speech = batch["wav"]  # [B, T]
        x_mask = batch["mask"]  # [B, n_frames]
        mel_feat = self._extract_mel_feature(speech)
        target_len = min(mel_feat.shape[1], x_mask.shape[1])

        cond_feat, cond_code = None, None

        sr = self.cfg.model.cond_sample_rate
        wavs = batch[f"wav_{sr}"]
        wav_lens = batch[f"wav_{sr}_len"]

        if self.cfg.model.cond_type == "hubert_features":
            cond_feat = self._extract_hubert_feature(wavs, wav_lens)  # [B, T, D]
            # TODO: This is dangerous, need to check the time resolution between cond_feat and mel
            target_len = min(target_len, cond_feat.shape[1])
            cond_feat = cond_feat[:, :target_len]

        elif self.cfg.model.cond_type == "hubert_codec":
            cond_code = self._extract_hubert_codec(wavs, wav_lens)  # [B, T]
            target_len = min(target_len, cond_code.shape[1])
            cond_code = cond_code[:, :target_len]

        mel_feat = mel_feat[:, :target_len]
        x_mask = x_mask[:, :target_len]

        # ## Debug ##
        # print("mel_feat: ", mel_feat.shape, mel_feat)
        # print("x_mask: ", x_mask.shape)
        # if cond_feat is not None:
        #     print("cond_feat: ", cond_feat.shape, cond_feat)
        # if cond_code is not None:
        #     print(
        #         "cond_code: ",
        #         cond_code.shape,
        #         cond_code,
        #         "max: ",
        #         cond_code.max(),
        #         "min: ",
        #         cond_code.min(),
        #     )

        torch.cuda.empty_cache()

        noise, x, flow_pred, final_mask, prompt_len = self.model(
            x=mel_feat,
            x_mask=x_mask,
            cond_code=cond_code,
            cond_feature=cond_feat,
        )

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

        return (total_loss.item(), train_losses, train_stats)
