# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random

import numpy as np

from torch.nn import functional as F

from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.vocoders.vocoder_dataset import VocoderDataset


class GANVocoderDataset(VocoderDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """
        super().__init__(cfg, dataset, is_valid)

        eval_index = random.randint(0, len(self.metadata) - 1)
        eval_utt_info = self.metadata[eval_index]
        eval_utt = "{}_{}".format(eval_utt_info["Dataset"], eval_utt_info["Uid"])
        self.eval_audio = np.load(self.utt2audio_path[eval_utt])
        if cfg.preprocess.use_mel:
            self.eval_mel = np.load(self.utt2mel_path[eval_utt])
        if cfg.preprocess.use_frame_pitch:
            self.eval_pitch = np.load(self.utt2frame_pitch_path[eval_utt])

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        if self.cfg.preprocess.use_mel:
            mel = np.load(self.utt2mel_path[utt])
            assert mel.shape[0] == self.cfg.preprocess.n_mel

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]

            if single_feature["target_len"] <= self.cfg.preprocess.cut_mel_frame:
                mel = np.pad(
                    mel,
                    ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                    mode="constant",
                )
            else:
                if "start" not in single_feature.keys():
                    start = random.randint(
                        0, mel.shape[-1] - self.cfg.preprocess.cut_mel_frame
                    )
                    end = start + self.cfg.preprocess.cut_mel_frame
                    single_feature["start"] = start
                    single_feature["end"] = end
                mel = mel[:, single_feature["start"] : single_feature["end"]]
            single_feature["mel"] = mel

        if self.cfg.preprocess.use_frame_pitch:
            frame_pitch = np.load(self.utt2frame_pitch_path[utt])
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_pitch)
            aligned_frame_pitch = align_length(
                frame_pitch, single_feature["target_len"]
            )

            if single_feature["target_len"] <= self.cfg.preprocess.cut_mel_frame:
                aligned_frame_pitch = np.pad(
                    aligned_frame_pitch,
                    (
                        (
                            0,
                            self.cfg.preprocess.cut_mel_frame
                            * self.cfg.preprocess.hop_size
                            - audio.shape[-1],
                        )
                    ),
                    mode="constant",
                )
            else:
                if "start" not in single_feature.keys():
                    start = random.randint(
                        0,
                        aligned_frame_pitch.shape[-1]
                        - self.cfg.preprocess.cut_mel_frame,
                    )
                    end = start + self.cfg.preprocess.cut_mel_frame
                    single_feature["start"] = start
                    single_feature["end"] = end
                aligned_frame_pitch = aligned_frame_pitch[
                    single_feature["start"] : single_feature["end"]
                ]
            single_feature["frame_pitch"] = aligned_frame_pitch

        if self.cfg.preprocess.use_audio:
            audio = np.load(self.utt2audio_path[utt])

            assert "target_len" in single_feature.keys()

            if (
                audio.shape[-1]
                <= self.cfg.preprocess.cut_mel_frame * self.cfg.preprocess.hop_size
            ):
                audio = np.pad(
                    audio,
                    (
                        (
                            0,
                            self.cfg.preprocess.cut_mel_frame
                            * self.cfg.preprocess.hop_size
                            - audio.shape[-1],
                        )
                    ),
                    mode="constant",
                )
            else:
                if "start" not in single_feature.keys():
                    audio = audio[
                        0 : self.cfg.preprocess.cut_mel_frame
                        * self.cfg.preprocess.hop_size
                    ]
                else:
                    audio = audio[
                        single_feature["start"]
                        * self.cfg.preprocess.hop_size : single_feature["end"]
                        * self.cfg.preprocess.hop_size,
                    ]
            single_feature["audio"] = audio

        if self.cfg.preprocess.use_amplitude_phase:
            logamp = np.load(self.utt2logamp_path[utt])
            pha = np.load(self.utt2pha_path[utt])
            rea = np.load(self.utt2rea_path[utt])
            imag = np.load(self.utt2imag_path[utt])

            assert "target_len" in single_feature.keys()

            if single_feature["target_len"] <= self.cfg.preprocess.cut_mel_frame:
                logamp = np.pad(
                    logamp,
                    ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                    mode="constant",
                )
                pha = np.pad(
                    pha,
                    ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                    mode="constant",
                )
                rea = np.pad(
                    rea,
                    ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                    mode="constant",
                )
                imag = np.pad(
                    imag,
                    ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                    mode="constant",
                )
            else:
                logamp = logamp[:, single_feature["start"] : single_feature["end"]]
                pha = pha[:, single_feature["start"] : single_feature["end"]]
                rea = rea[:, single_feature["start"] : single_feature["end"]]
                imag = imag[:, single_feature["start"] : single_feature["end"]]
            single_feature["logamp"] = logamp
            single_feature["pha"] = pha
            single_feature["rea"] = rea
            single_feature["imag"] = imag

        return single_feature


class GANVocoderCollator(object):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, n_mels, frame]
        # frame_pitch: [b, frame]
        # audios: [b, frame * hop_size]

        for key in batch[0].keys():
            if key in ["target_len", "start", "end"]:
                continue
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
