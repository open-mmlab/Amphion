# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import torch
import torch.nn as nn
import numpy as np

from models.base.new_trainer import BaseTrainer
from models.svc.base.svc_dataset import (
    SVCOfflineCollator,
    SVCOfflineDataset,
    SVCOnlineCollator,
    SVCOnlineDataset,
)
from processors.audio_features_extractor import AudioFeaturesExtractor
from processors.acoustic_extractor import cal_normalized_mel, load_mel_extrema

EPS = 1.0e-12


class SVCTrainer(BaseTrainer):
    r"""The base trainer for all SVC models. It inherits from BaseTrainer and implements
    ``build_criterion``, ``_build_dataset`` and ``_build_singer_lut`` methods. You can inherit from this
    class, and implement ``_build_model``, ``_forward_step``.
    """

    def __init__(self, args=None, cfg=None):
        self.args = args
        self.cfg = cfg

        self._init_accelerator()

        # Only for SVC tasks
        with self.accelerator.main_process_first():
            self.singers = self._build_singer_lut()

        # Super init
        BaseTrainer.__init__(self, args, cfg)

        # Only for SVC tasks
        self.task_type = "SVC"
        self.logger.info("Task type: {}".format(self.task_type))

    ### Following are methods only for SVC tasks ###
    def _build_dataset(self):
        self.online_features_extraction = (
            self.cfg.preprocess.features_extraction_mode == "online"
        )

        if not self.online_features_extraction:
            return SVCOfflineDataset, SVCOfflineCollator
        else:
            self.audio_features_extractor = AudioFeaturesExtractor(self.cfg)
            return SVCOnlineDataset, SVCOnlineCollator

    def _extract_svc_features(self, batch):
        """
        Features extraction during training

        Batch:
            wav: (B, T)
            wav_len: (B)
            target_len: (B)
            mask: (B, n_frames, 1)
            spk_id: (B, 1)

            wav_{sr}: (B, T)
            wav_{sr}_len: (B)

        Added elements when output:
            mel: (B, n_frames, n_mels)
            frame_pitch: (B, n_frames)
            frame_uv: (B, n_frames)
            frame_energy: (B, n_frames)
            frame_{content}: (B, n_frames, D)
        """

        padded_n_frames = torch.max(batch["target_len"])
        final_n_frames = padded_n_frames

        ### Mel Spectrogram ###
        if self.cfg.preprocess.use_mel:
            # (B, n_mels, n_frames)
            raw_mel = self.audio_features_extractor.get_mel_spectrogram(batch["wav"])
            if self.cfg.preprocess.use_min_max_norm_mel:
                # TODO: Change the hard code

                # Using the empirical mel extrema to denormalize
                if not hasattr(self, "mel_extrema"):
                    # (n_mels)
                    m, M = load_mel_extrema(self.cfg.preprocess, "vctk")
                    # (1, n_mels, 1)
                    m = (
                        torch.as_tensor(m, device=raw_mel.device)
                        .unsqueeze(0)
                        .unsqueeze(-1)
                    )
                    M = (
                        torch.as_tensor(M, device=raw_mel.device)
                        .unsqueeze(0)
                        .unsqueeze(-1)
                    )
                    self.mel_extrema = m, M

                m, M = self.mel_extrema
                mel = (raw_mel - m) / (M - m + EPS) * 2 - 1

            else:
                mel = raw_mel

            final_n_frames = min(final_n_frames, mel.size(-1))

            # (B, n_frames, n_mels)
            batch["mel"] = mel.transpose(1, 2)
        else:
            raw_mel = None

        ### F0 ###
        if self.cfg.preprocess.use_frame_pitch:
            # (B, n_frames)
            raw_f0, raw_uv = self.audio_features_extractor.get_f0(
                batch["wav"],
                wav_lens=batch["wav_len"],
                use_interpolate=self.cfg.preprocess.use_interpolation_for_uv,
                return_uv=True,
            )
            final_n_frames = min(final_n_frames, raw_f0.size(-1))
            batch["frame_pitch"] = raw_f0

            if self.cfg.preprocess.use_uv:
                batch["frame_uv"] = raw_uv

        ### Energy ###
        if self.cfg.preprocess.use_frame_energy:
            # (B, n_frames)
            raw_energy = self.audio_features_extractor.get_energy(
                batch["wav"], mel_spec=raw_mel
            )
            final_n_frames = min(final_n_frames, raw_energy.size(-1))
            batch["frame_energy"] = raw_energy

        ### Semantic Features ###
        if self.cfg.model.condition_encoder.use_whisper:
            # (B, n_frames, D)
            whisper_feats = self.audio_features_extractor.get_whisper_features(
                wavs=batch["wav_{}".format(self.cfg.preprocess.whisper_sample_rate)],
                target_frame_len=padded_n_frames,
            )
            final_n_frames = min(final_n_frames, whisper_feats.size(1))
            batch["whisper_feat"] = whisper_feats

        if self.cfg.model.condition_encoder.use_contentvec:
            # (B, n_frames, D)
            contentvec_feats = self.audio_features_extractor.get_contentvec_features(
                wavs=batch["wav_{}".format(self.cfg.preprocess.contentvec_sample_rate)],
                target_frame_len=padded_n_frames,
            )
            final_n_frames = min(final_n_frames, contentvec_feats.size(1))
            batch["contentvec_feat"] = contentvec_feats

        if self.cfg.model.condition_encoder.use_wenet:
            # (B, n_frames, D)
            wenet_feats = self.audio_features_extractor.get_wenet_features(
                wavs=batch["wav_{}".format(self.cfg.preprocess.wenet_sample_rate)],
                target_frame_len=padded_n_frames,
                wav_lens=batch[
                    "wav_{}_len".format(self.cfg.preprocess.wenet_sample_rate)
                ],
            )
            final_n_frames = min(final_n_frames, wenet_feats.size(1))
            batch["wenet_feat"] = wenet_feats

        ### Align all the audio features to the same frame length ###
        frame_level_features = [
            "mask",
            "mel",
            "frame_pitch",
            "frame_uv",
            "frame_energy",
            "whisper_feat",
            "contentvec_feat",
            "wenet_feat",
        ]
        for k in frame_level_features:
            if k in batch:
                # (B, n_frames, ...)
                batch[k] = batch[k][:, :final_n_frames].contiguous()

        return batch

    @staticmethod
    def _build_criterion():
        criterion = nn.MSELoss(reduction="none")
        return criterion

    @staticmethod
    def _compute_loss(criterion, y_pred, y_gt, loss_mask):
        """
        Args:
            criterion: MSELoss(reduction='none')
            y_pred, y_gt: (B, seq_len, D)
            loss_mask: (B, seq_len, 1)
        Returns:
            loss: Tensor of shape []
        """

        # (B, seq_len, D)
        loss = criterion(y_pred, y_gt)
        # expand loss_mask to (B, seq_len, D)
        loss_mask = loss_mask.repeat(1, 1, loss.shape[-1])

        loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)
        return loss

    def _save_auxiliary_states(self):
        """
        To save the singer's look-up table in the checkpoint saving path
        """
        with open(
            os.path.join(self.tmp_checkpoint_save_path, self.cfg.preprocess.spk2id),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.singers, f, indent=4, ensure_ascii=False)

    def _build_singer_lut(self):
        resumed_singer_path = None
        if self.args.resume_from_ckpt_path and self.args.resume_from_ckpt_path != "":
            resumed_singer_path = os.path.join(
                self.args.resume_from_ckpt_path, self.cfg.preprocess.spk2id
            )
        if os.path.exists(os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)):
            resumed_singer_path = os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)

        if resumed_singer_path:
            with open(resumed_singer_path, "r") as f:
                singers = json.load(f)
        else:
            singers = dict()

        for dataset in self.cfg.dataset:
            singer_lut_path = os.path.join(
                self.cfg.preprocess.processed_dir, dataset, self.cfg.preprocess.spk2id
            )
            with open(singer_lut_path, "r") as singer_lut_path:
                singer_lut = json.load(singer_lut_path)
            for singer in singer_lut.keys():
                if singer not in singers:
                    singers[singer] = len(singers)

        with open(
            os.path.join(self.exp_dir, self.cfg.preprocess.spk2id), "w"
        ) as singer_file:
            json.dump(singers, singer_file, indent=4, ensure_ascii=False)
        print(
            "singers have been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            )
        )
        return singers
