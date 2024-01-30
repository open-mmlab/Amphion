# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.base.base_dataset import (
    BaseCollator,
    BaseDataset,
    BaseTestDataset,
    BaseTestCollator,
)
import librosa


class AutoencoderKLDataset(BaseDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        BaseDataset.__init__(self, cfg, dataset, is_valid=is_valid)

        cfg = self.cfg

        # utt2melspec
        if cfg.preprocess.use_melspec:
            self.utt2melspec_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2melspec_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.melspec_dir,
                    uid + ".npy",
                )

        # utt2wav
        if cfg.preprocess.use_wav:
            self.utt2wav_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2wav_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.wav_dir,
                    uid + ".wav",
                )

    def __getitem__(self, index):
        # melspec: (n_mels, T)
        # wav: (T,)

        single_feature = BaseDataset.__getitem__(self, index)

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        if self.cfg.preprocess.use_melspec:
            single_feature["melspec"] = np.load(self.utt2melspec_path[utt])

        if self.cfg.preprocess.use_wav:
            wav, sr = librosa.load(
                self.utt2wav_path[utt], sr=16000
            )  # hard coding for 16KHz...
            single_feature["wav"] = wav

        return single_feature

    def __len__(self):
        return len(self.metadata)

    def __len__(self):
        return len(self.metadata)


class AutoencoderKLCollator(BaseCollator):
    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)

    def __call__(self, batch):
        # mel: (B, n_mels, T)
        # wav (option): (B, T)

        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "melspec":
                packed_batch_features["melspec"] = torch.from_numpy(
                    np.array([b["melspec"][:, :624] for b in batch])
                )

            if key == "wav":
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features


class AutoencoderKLTestDataset(BaseTestDataset): ...


class AutoencoderKLTestCollator(BaseTestCollator): ...
