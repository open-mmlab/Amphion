# Copyright (c) 2023 Amphion.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""

This module aims to be an entrance that integrates all the "audio" features extraction functions.

The common audio features include:
1. Acoustic features such as Mel Spectrogram, F0, Energy, etc.
2. Content features such as phonetic posteriorgrams (PPG) and bottleneck features (BNF) from pretrained

Note: 
All the features extraction are designed to utilize GPU to the maximum extent, which can ease the on-the-fly extraction for large-scale dataset.

"""

import torch

from utils.mel import extract_mel_features
from utils.f0 import get_f0 as extract_f0_features
from processors.content_extractor import (
    WhisperExtractor,
    ContentvecExtractor,
    WenetExtractor,
)


class AudioFeaturesExtractor:
    def __init__(self, cfg):
        """
        Args:
            cfg: Amphion config that would be used to specify the processing parameters
        """
        self.cfg = cfg

    def get_mel_spectrogram(self, wavs):
        """Get Mel Spectrogram Features

        Args:
            wavs: Tensor whose shape is (B, T)

        Returns:
            Tensor whose shape is (B, n_mels, n_frames)
        """
        return extract_mel_features(y=wavs, cfg=self.cfg.preprocess)

    def get_f0(self, wavs):
        """Get F0 Features

        Args:
            wavs: Tensor whose shape is (B, T)

        Returns:
            Tensor whose shape is (B, n_frames)
        """
        f0s = []
        for w in wavs:
            # TODO: use numpy to extract
            f0s.append(extract_f0_features(w, self.cfg.preprocess))

        f0s = torch.stack(f0s, dim=0)

        # TODO: 融入interpolate的判断、return v/uv的判断
        return f0s

    def get_energy(self, wavs, mel_spec=None):
        """Get Energy Features

        Args:
            wavs: Tensor whose shape is (B, T)
            mel_spec: Tensor whose shape is (B, n_mels, n_frames)

        Returns:
            Tensor whose shape is (B, n_frames)
        """
        if mel_spec is None:
            mel_spec = self.get_mel_spectrogram(wavs)

        return (mel_spec.exp() ** 2).sum(dim=1).sqrt()

    def get_whisper_features(self, wavs, target_frame_len):
        """Get Whisper Features

        Args:
            wavs: Tensor whose shape is (B, T)
            target_frame_len: int

        Returns:
            Tensor whose shape is (B, target_frame_len, D)
        """
        if not hasattr(self, "whisper_extractor"):
            self.whisper_extractor = WhisperExtractor(self.cfg)
            self.whisper_extractor.load_model()

        whisper_feats = self.whisper_extractor.extract_content_features(wavs)
        whisper_feats = self.whisper_extractor.ReTrans(whisper_feats, target_frame_len)
        return whisper_feats

    def get_contentvec_features(self, wavs, target_frame_len):
        """Get ContentVec Features

        Args:
            wavs: Tensor whose shape is (B, T)
            target_frame_len: int

        Returns:
            Tensor whose shape is (B, target_frame_len, D)
        """
        if not hasattr(self, "contentvec_extractor"):
            self.contentvec_extractor = ContentvecExtractor(self.cfg)
            self.contentvec_extractor.load_model()

        contentvec_feats = self.contentvec_extractor.extract_content_features(wavs)
        contentvec_feats = self.contentvec_extractor.ReTrans(
            contentvec_feats, target_frame_len
        )
        return contentvec_feats

    def get_wenet_features(self, wavs, target_frame_len, wav_lens=None):
        """Get WeNet Features

        Args:
            wavs: Tensor whose shape is (B, T)
            target_frame_len: int
            wav_lens: Tensor whose shape is (B)

        Returns:
            Tensor whose shape is (B, target_frame_len, D)
        """
        if not hasattr(self, "wenet_extractor"):
            self.wenet_extractor = WenetExtractor(self.cfg)
            self.wenet_extractor.load_model()

        wenet_feats = self.wenet_extractor.extract_content_features(wavs, lens=wav_lens)
        wenet_feats = self.wenet_extractor.ReTrans(wenet_feats, target_frame_len)
        return wenet_feats
