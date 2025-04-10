# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
import torch
import numpy as np
import math
import torchaudio
import yaml

from models.base.base_trainer import BaseTrainer
from models.codec.melvqgan.melspec import MelSpectrogram
from models.vc.autoregressive_transformer.ar_model import AutoregressiveTransformer
from models.codec.kmeans.repcodec_model import RepCodec
from models.codec.vevo.vevo_repcodec import VevoRepCodec
from models.vc.base.vc_emilia_dataset import VCEmiliaDataset, VCCollator

import safetensors


class AutoregressiveTransformerTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(AutoregressiveTransformerTrainer, self).__init__(args, cfg)

        # setup input model (such as content tokens)
        self._build_input_model()

        # setup output model (such as content-style tokens)
        self._build_output_model()

        # Global Style Encoder
        if self.cfg.model.autoregressive_transformer.use_global_style_encoder:
            self._build_mel_model()

    def _build_model(self):
        model = AutoregressiveTransformer(cfg=self.cfg.model.autoregressive_transformer)
        return model

    @torch.no_grad()
    def _extract_hubert_feature(self, wavs, wav_lens=None, output_layer=18):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            feats: [B, T, D]
            feat_lengths: [B]
        """
        feats, feat_lengths = self.hubert.extract_features(
            wavs, lengths=wav_lens, num_layers=output_layer
        )
        feats = feats[-1]
        return feats, feat_lengths

    def get_reduced_sequence(self, token_seq, n_gram=1):
        """
        Args:
            token_seq: (T,)
        Returns:
            reduced_token_seq: (T')
            reduced_token_seq_len: T'
        """
        # # (T,) Note that the first token should always be selected
        # diff = torch.diff(
        #     torch.cat([token_seq.new_tensor([token_seq[0] - 1]), token_seq])
        # )
        # # (T',)
        # key_indices = torch.nonzero(diff, as_tuple=True)[0]
        # # (T',)
        # reduced_token_seq = token_seq[key_indices]

        n_gram_seq = token_seq.unfold(0, n_gram, 1)
        mask = torch.all(n_gram_seq[1:] != n_gram_seq[:-1], dim=1)
        reduced_token_seq = torch.cat(
            (n_gram_seq[0, :n_gram], n_gram_seq[1:, -1][mask])
        )
        return reduced_token_seq, len(reduced_token_seq)

    @torch.no_grad()
    def _extract_hubert_codec(
        self,
        vqvae_model,
        wavs,
        wav_lens=None,
        output_layer=18,
        token_type="hubert_codec",
        duration_reduction=False,
        duration_reduction_n_gram=1,
        feats=None,
        feat_lengths=None,
    ):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            codecs: [B, T]
            codec_masks: [B, T]
        """
        # Extract features and normalize
        if feats is None:
            feats, feat_lengths = self._extract_hubert_feature(
                wavs, wav_lens, output_layer
            )

        if token_type == "hubert_codec":
            feats = (feats - self.feat_norm_mean.to(feats)) / self.feat_norm_std.to(
                feats
            )
            codecs, _ = vqvae_model.quantize(feats)  # (B, T)
        elif token_type == "hubert_vevo_codec":
            x = vqvae_model.encoder(feats.transpose(1, 2))
            z = vqvae_model.projector(x)
            _, idx = vqvae_model.quantizer.codebook.forward_index(z.transpose(2, 1))
            codecs = idx[0]  # (B, T)
        else:
            raise ValueError("Invalid token_type")

        if not duration_reduction:
            T = codecs.shape[1]
            arange_tensor = torch.arange(T).expand(codecs.shape[0], T).to(codecs)
            codec_masks = (
                arange_tensor < feat_lengths.unsqueeze(-1)
            ).int()  # 1 means valid
            return codecs, codec_masks

        else:
            reduced_codecs = []
            reduced_masks = []
            for i, token_seq_len in enumerate(feat_lengths):
                token_seq = codecs[i, :token_seq_len]
                reduced_token_seq, reduced_token_seq_len = self.get_reduced_sequence(
                    token_seq, n_gram=duration_reduction_n_gram
                )
                # print("Raw: ", token_seq[:30], token_seq_len)
                # print("Reduced: ", reduced_token_seq[:30], reduced_token_seq_len)
                reduced_codecs.append(reduced_token_seq)
                reduced_masks.append(
                    torch.ones(reduced_token_seq_len, dtype=torch.int).to(codecs)
                )

            reduced_codecs = torch.nn.utils.rnn.pad_sequence(
                reduced_codecs, batch_first=True, padding_value=0
            )
            reduced_masks = torch.nn.utils.rnn.pad_sequence(
                reduced_masks, batch_first=True, padding_value=0
            )
            return reduced_codecs, reduced_masks

    def _random_mask_codec(
        self, codecs, codec_masks, do_mask_prob, max_mask_ratio, mask_value
    ):
        """
        Args:
            codecs: [B, T]
            codec_masks: [B, T]
            do_mask_prob: float
            max_mask_ratio: float
            mask_value: int
        Returns:
            masked_codecs: [B, T]
        """
        if random.random() > do_mask_prob:
            return codecs

        ratio = random.random() * max_mask_ratio
        rand_mask = (torch.rand_like(codecs.float(), device=codecs.device) < ratio) & (
            codec_masks == 1
        )
        masked_codecs = codecs.masked_fill(rand_mask, mask_value)
        return masked_codecs

    def _build_hubert_model(self):
        if not hasattr(self, "hubert"):
            bundle = torchaudio.pipelines.HUBERT_LARGE
            self.hubert = bundle.get_model()
            self.hubert.eval()
            self.hubert.to(self.accelerator.device)

            # Features normalization
            stat = np.load(self.cfg.model.representation_stat_mean_var_path)
            self.feat_norm_mean = torch.tensor(stat["mean"]).to(self.accelerator.device)
            self.feat_norm_std = torch.tensor(stat["std"]).to(self.accelerator.device)

    def _build_input_model(self):
        if self.cfg.model.vc_input_token_type == "hubert_codec":
            self._build_hubert_model()

            self.vqvae_input = RepCodec(cfg=self.cfg.model.input_repcodec)
            self.vqvae_input.eval()

            pretrained_path = self.cfg.model.input_repcodec.pretrained_path
            if ".bin" in pretrained_path:
                self.vqvae_input.load_state_dict(torch.load(pretrained_path))
            elif ".safetensors" in pretrained_path:
                safetensors.torch.load_model(self.vqvae_input, pretrained_path)
            self.vqvae_input.to(self.accelerator.device)

        elif self.cfg.model.vc_input_token_type == "hubert_vevo_codec":
            self._build_hubert_model()

            with open(self.cfg.model.input_repcodec.config_path) as fp:
                conf = yaml.load(fp, Loader=yaml.FullLoader)

            self.vqvae_input = VevoRepCodec(**conf)
            self.vqvae_input.quantizer.initial()
            self.vqvae_input.eval()

            pretrained_path = self.cfg.model.input_repcodec.pretrained_path
            if ".pkl" in pretrained_path:
                self.vqvae_input.load_state_dict(
                    torch.load(pretrained_path, map_location="cpu")["model"]["repcodec"]
                )
            elif ".safetensors" in pretrained_path:
                safetensors.torch.load_model(self.vqvae_input, pretrained_path)

            self.vqvae_input.to(self.accelerator.device)

    def _build_output_model(self):
        if self.cfg.model.output_token_type == "hubert_codec":
            self._build_hubert_model()

            # VQ-VAE Tokenizer
            self.vqvae_output = RepCodec(cfg=self.cfg.model.output_repcodec)
            self.vqvae_output.eval()

            pretrained_path = self.cfg.model.output_repcodec.pretrained_path
            if ".bin" in pretrained_path:
                self.vqvae_output.load_state_dict(torch.load(pretrained_path))
            elif ".safetensors" in pretrained_path:
                safetensors.torch.load_model(self.vqvae_output, pretrained_path)
            self.vqvae_output.to(self.accelerator.device)

    def _build_mel_model(self):
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

    def _build_dataset(self):
        return VCEmiliaDataset, VCCollator

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        speech = batch["wav"]  # [B, T]
        x_mask = batch["mask"]  # [B, n_frames]
        if self.cfg.model.autoregressive_transformer.use_global_style_encoder:
            mel_feat = self._extract_mel_feature(speech)
        else:
            mel_feat = None

        sr = 16000
        wavs = batch[f"wav_{sr}"]
        wav_lens = batch[f"wav_{sr}_len"]

        # [B, T]
        feats, feat_lengths = self._extract_hubert_feature(wavs, wav_lens)
        output_ids, output_mask = self._extract_hubert_codec(
            vqvae_model=self.vqvae_output,
            wavs=wavs,
            wav_lens=wav_lens,
            token_type=self.cfg.model.output_token_type,
            duration_reduction=False,
            feats=feats,
            feat_lengths=feat_lengths,
        )

        if self.cfg.model.train_both_vc_and_tts:
            # Randomly choose to train VC or TTS
            if random.random() < 0.5:
                task = "TTS"
                input_ids = batch["phone_id"]
                input_mask = batch["phone_mask"]
            else:
                task = "VC"
                input_ids, input_mask = self._extract_hubert_codec(
                    vqvae_model=self.vqvae_input,
                    wavs=wavs,
                    wav_lens=wav_lens,
                    token_type=self.cfg.model.vc_input_token_type,
                    duration_reduction=True,
                    duration_reduction_n_gram=getattr(
                        self.cfg.model, "vc_input_reduced_n_gram", 1
                    ),
                    feats=feats,
                    feat_lengths=feat_lengths,
                )

                if getattr(self.cfg.model, "vc_random_mask_input_prob", -1) > 0:
                    input_ids = self._random_mask_codec(
                        codecs=input_ids,
                        codec_masks=input_mask,
                        do_mask_prob=self.cfg.model.vc_random_mask_input_prob,
                        max_mask_ratio=self.cfg.model.vc_random_mask_input_max_ratio,
                        mask_value=self.cfg.model.vc_input_vocab_size,
                    )

                # [Important] When traing both VC and TTS, the VC's input_ids should be shifted, since Llama use a unified codebook
                input_ids = (
                    input_ids + self.cfg.model.tts_input_vocab_size
                ) * input_mask

        elif getattr(self.cfg.model, "train_vc", False):
            task = "VC"
            input_ids, input_mask = self._extract_hubert_codec(
                vqvae_model=self.vqvae_input,
                wavs=wavs,
                wav_lens=wav_lens,
                token_type=self.cfg.model.vc_input_token_type,
                duration_reduction=True,
                duration_reduction_n_gram=getattr(
                    self.cfg.model, "vc_input_reduced_n_gram", 1
                ),
                feats=feats,
                feat_lengths=feat_lengths,
            )

            if getattr(self.cfg.model, "vc_random_mask_input_prob", -1) > 0:
                input_ids = self._random_mask_codec(
                    codecs=input_ids,
                    codec_masks=input_mask,
                    do_mask_prob=self.cfg.model.vc_random_mask_input_prob,
                    max_mask_ratio=self.cfg.model.vc_random_mask_input_max_ratio,
                    mask_value=self.cfg.model.vc_input_vocab_size,
                )

        elif getattr(self.cfg.model, "train_tts", False):
            task = "TTS"
            input_ids = batch["phone_id"]
            input_mask = batch["phone_mask"]

        torch.cuda.empty_cache()

        # ## Debug ##
        # print("Task: ", task)
        # print(
        #     "input_ids: ",
        #     input_ids.shape,
        #     input_ids,
        #     "max: ",
        #     input_ids.max(),
        #     "min: ",
        #     input_ids.min(),
        # )
        # print("input_mask: ", input_mask.shape, input_mask)
        # print(
        #     "output_ids: ",
        #     output_ids.shape,
        #     output_ids,
        #     "max: ",
        #     output_ids.max(),
        #     "min: ",
        #     output_ids.min(),
        # )
        # print("output_mask: ", output_mask.shape, output_mask)

        out = self.model(
            input_ids,
            input_mask,
            output_ids,
            output_mask,
            mels=mel_feat,
            mels_mask=x_mask,
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

        train_losses["batch_size"] = wavs.shape[0]
        train_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return (total_loss.item(), train_losses, train_stats)
