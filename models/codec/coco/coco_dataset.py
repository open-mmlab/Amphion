# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import json
import parselmouth
import numpy as np
import librosa
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from librosa.feature import chroma_stft

from utils.f0 import f0_to_coarse, interpolate
from models.vc.base.vc_emilia_dataset import VCEmiliaDataset

logger = logging.getLogger(__name__)


class CocoDataset(VCEmiliaDataset):
    def __init__(self, cfg):
        super(CocoDataset, self).__init__(cfg=cfg)
        self.emilia_size = len(self.wav_paths)

        self.singnet_ratio = self.dataset_ratio_dict["singnet"]
        self.init_for_singnet()

    def init_for_singnet(self):
        assert "singnet_path" in self.cfg, "singnet_path is not in the config"
        singnet_path = self.cfg.singnet_path

        with open(singnet_path, "r") as f:
            metadata = json.load(f)

        # Scale according to the ratio
        self.singnet_metadata = metadata * self.singnet_ratio
        self.singnet_size = len(self.singnet_metadata)

        self.singnet_wav_paths = [item["Path"] for item in self.singnet_metadata]
        self.singnet_wav_paths_index2duration = {
            idx: item["Duration"] for idx, item in enumerate(self.singnet_metadata)
        }

        # For batch sampler
        self.index2num_frames.extend(
            [item["Duration"] * 50 for item in self.singnet_metadata]
        )
        assert len(self.index2num_frames) == self.emilia_size + self.singnet_size
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.index2num_frames)),
                key=lambda k: self.index2num_frames[k],
            )
        )

    def __len__(self):
        return self.wav_paths.__len__() + self.singnet_wav_paths.__len__()

    def get_num_frames(self, index):
        if index < self.emilia_size:
            return self.wav_path_index2duration[index] * 50
        else:
            return self.singnet_wav_paths_index2duration[index - self.emilia_size] * 50

    def extract_f0(self, speech, speech_frames=None):
        """
        Args:
            speech: (T,) at self.sample_rate
            speech_frames: int
        Returns:
            f0: (T,)
        """
        f0 = (
            parselmouth.Sound(speech, self.sample_rate)
            .to_pitch_ac(
                time_step=self.cfg.preprocess.hop_size / self.sample_rate,
                voicing_threshold=0.6,
                pitch_floor=self.cfg.preprocess.f0_fmin,
                pitch_ceiling=self.cfg.preprocess.f0_fmax,
            )
            .selected_array["frequency"]
        )
        f0, _ = interpolate(f0)

        if speech_frames is not None:
            if f0.shape[0] < speech_frames:
                f0 = np.pad(f0, (0, speech_frames - f0.shape[0]), mode="edge")
            else:
                f0 = f0[:speech_frames]

        return f0

    def get_tone_height(self, interpolated_f0):
        m, M = self.cfg.preprocess.f0_fmin, self.cfg.preprocess.f0_fmax
        med = np.median(interpolated_f0)  # note that there is no 0 in interpolated_f0
        tone_height = (med - m) / (M - m)
        return tone_height

    def get_random_shifted_f0_bins(self, f0, augmentor_spk):
        """
        Args:
            f0: (T,), Hz
            augmentor_spk: str
        Returns:
            shifted_f0_bins: (T,), quantized to 256 bins
        """
        # Random choose a "reasonal" shift step
        target_hz = self.augmentor_speaker_dict[augmentor_spk]["f0_median"]
        src_hz = np.median(f0[f0 > 0])
        autoshift_steps = int(12 * np.log2(target_hz / src_hz))

        var = np.random.randint(-3, 4)  # [-3, 3]
        f0_shift_n_steps = autoshift_steps + var

        shifted_f0 = f0 * 2 ** (f0_shift_n_steps / 12)

        # Hz to bins
        shifted_f0_bins = f0_to_coarse(
            shifted_f0,
            pitch_bin=self.cfg.model.augmentor.f0_bins,
            f0_min=self.cfg.preprocess.f0_fmin,
            f0_max=self.cfg.preprocess.f0_fmax,
        )
        return shifted_f0_bins

    def get_chromagram(self, speech, speech_frames):
        # [24, T] -> [T, 24]
        chromagram = chroma_stft(
            y=speech,
            sr=self.sample_rate,
            n_fft=self.cfg.preprocess.n_fft,
            hop_length=self.cfg.preprocess.hop_size,
            win_length=self.cfg.preprocess.win_size,
            n_chroma=24,
        ).T

        if chromagram.shape[0] < speech_frames:
            chromagram = np.pad(
                chromagram, (0, speech_frames - chromagram.shape[0]), mode="edge"
            )
        else:
            chromagram = chromagram[:speech_frames]

        return chromagram

    def get_item_for_singnet(self, idx):
        item = self.singnet_metadata[idx]

        single_features = dict()
        try:
            speech, _ = librosa.load(item["Path"], sr=self.sample_rate)
        except:
            raise Exception("Failed to load file {}".format(item["Path"]))

        # pad the speech to the multiple of hop_size
        speech = np.pad(
            speech,
            (
                0,
                self.cfg.preprocess.hop_size
                - len(speech) % self.cfg.preprocess.hop_size,
            ),
            mode="constant",
        )

        # For all the sample rates
        for tgt_sr in self.all_sample_rates:
            if tgt_sr != self.sample_rate:
                assert tgt_sr < self.sample_rate
                tgt_speech = librosa.resample(
                    speech, orig_sr=self.sample_rate, target_sr=tgt_sr
                )
            else:
                tgt_speech = speech
            single_features.update(
                {
                    f"wav_{tgt_sr}": tgt_speech,
                    f"wav_{tgt_sr}_len": len(tgt_speech),
                }
            )

        # [Note] Mask is (n_frames,) but not (T,)
        speech_frames = len(speech) // self.cfg.preprocess.hop_size
        mask = np.ones(speech_frames)

        single_features.update(
            {
                "wav": speech,
                "wav_len": len(speech),
                "mask": mask,
            }
        )

        ## Load phone using G2P ##
        if self.load_phone:
            try:
                phone_id = self.g2p(item["Text"], item["Language"])[1]
                phone_id = torch.tensor(np.array(phone_id), dtype=torch.long)
                phone_mask = np.ones(len(phone_id))

                single_features.update({"phone_id": phone_id, "phone_mask": phone_mask})
            except Exception as e:
                print(f"Error of loading phone in get_item_for_singnet: {e}")
                print(f"Item: {item}")
                raise e

        if self.load_wav_path:
            single_features.update({"wav_path": item["Path"]})

        return single_features

    def __getitem__(self, idx):
        """
        Returns:
            dict:
                wav: (T,)
                wav_len: int
                mask: (n_frames,)

                wav_{sr}: (T,)
                wav_{sr}_len: int

                phone_id: (n_phones,)
                phone_mask: (n_phones,)

                tone_height: float, range from 0 to 1
                chromagram: (T, 24)

                augmentor_spk_id: int
                augmentor_f0: (T,), quantized to 256 bins
        """
        try:
            if idx < self.emilia_size:
                single_features = super(CocoDataset, self).__getitem__(idx)
            else:
                single_features = self.get_item_for_singnet(idx - self.emilia_size)

            speech = single_features["wav_{}".format(self.sample_rate)]
            speech_frames = len(speech) // self.cfg.preprocess.hop_size

            if self.load_chromagram:

                # f0 = self.extract_f0(speech, speech_frames)  # Hz
                # tone_height = self.get_tone_height(f0)  # range from 0 to 1
                # single_features["tone_height"] = tone_height

                chromagram = self.get_chromagram(speech, speech_frames)
                single_features["chromagram"] = chromagram

        except Exception as e:
            logger.error(f"Error in __getitem__({idx}): {e}")
            p = (
                self.singnet_wav_paths[idx - self.emilia_size]
                if idx >= self.emilia_size
                else self.wav_paths[idx]
            )
            logger.error(f"Error for wav path: {p}")

            try:
                for k, v in single_features.items():
                    if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                        print(k, v.shape, v.min(), v.max())
                    else:
                        print(k, type(v), v)
            except Exception as e:
                pass

            return self.__getitem__(random.randint(0, len(self) - 1))

        return single_features


class CocoCollator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        """
        CocoDataset.__getitem__:
            wav: (T,)
            wav_len: int
            mask: (n_frames,)

            wav_{sr}: (T,)
            wav_{sr}_len: int

            phone_id: (n_phones,)
            phone_mask: (n_phones,)

            tone_height: float, range from 0 to 1
            chromagram: (T, 24)

            augmentor_spk_id: int
            augmentor_f0: (T,), quantized to 256 bins

        Returns:
            wav: (B, T), torch.float32
            wav_len: (B), torch.long
            mask: (B, n_frames), torch.float32

            wav_{sr}: (B, T)
            wav_{sr}_len: (B), torch.long

            phone_id: (B, n_phones), torch.long
            phone_mask: (B, n_phones), torch.float32

            tone_height: (B), torch.float32
            chromagram: (B, T, 24), torch.float32

            augmentor_spk_id: (B), torch.long
            augmentor_f0: (B, T), torch.long
        """

        packed_batch_features = dict()

        for key in batch[0].keys():
            if "_len" in key or "spk_id" in key:
                packed_batch_features[key] = torch.LongTensor([b[key] for b in batch])
            elif key == "tone_height":
                packed_batch_features[key] = torch.FloatTensor([b[key] for b in batch])
            elif key == "phone_id":
                packed_batch_features[key] = pad_sequence(
                    [utt[key].long() for utt in batch],
                    batch_first=True,
                    padding_value=1023,  # phone vocab size is 1024
                )
            elif key == "wav_path":
                packed_batch_features[key] = [b[key] for b in batch]
            else:
                packed_batch_features[key] = pad_sequence(
                    [torch.as_tensor(b[key]) for b in batch],
                    batch_first=True,
                    padding_value=0,
                )
        return packed_batch_features
