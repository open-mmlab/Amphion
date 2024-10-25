# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio
import json
import os
import numpy as np
import librosa
import whisper
from torch.nn.utils.rnn import pad_sequence


class TorchaudioDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, sr, accelerator=None, metadata=None):
        """
        Args:
            cfg: config
            dataset: dataset name

        """
        assert isinstance(dataset, str)

        self.sr = sr
        self.cfg = cfg

        if metadata is None:
            self.train_metadata_path = os.path.join(
                cfg.preprocess.processed_dir, dataset, cfg.preprocess.train_file
            )
            self.valid_metadata_path = os.path.join(
                cfg.preprocess.processed_dir, dataset, cfg.preprocess.valid_file
            )
            self.metadata = self.get_metadata()
        else:
            self.metadata = metadata

        if accelerator is not None:
            self.device = accelerator.device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def get_metadata(self):
        metadata = []
        with open(self.train_metadata_path, "r", encoding="utf-8") as t:
            metadata.extend(json.load(t))
        with open(self.valid_metadata_path, "r", encoding="utf-8") as v:
            metadata.extend(json.load(v))
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        utt_info = self.metadata[index]
        wav_path = utt_info["Path"]

        wav, sr = torchaudio.load(wav_path)

        # resample
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        # downmixing
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        assert wav.shape[0] == 1
        wav = wav.squeeze(0)
        # record the length of wav without padding
        length = wav.shape[0]
        # wav: (T)
        return utt_info, wav, length


class LibrosaDataset(TorchaudioDataset):
    def __init__(self, cfg, dataset, sr, accelerator=None, metadata=None):
        super().__init__(cfg, dataset, sr, accelerator, metadata)

    def __getitem__(self, index):
        utt_info = self.metadata[index]
        wav_path = utt_info["Path"]

        wav, _ = librosa.load(wav_path, sr=self.sr)
        # wav: (T)
        wav = torch.from_numpy(wav)

        # record the length of wav without padding
        length = wav.shape[0]
        return utt_info, wav, length


class FFmpegDataset(TorchaudioDataset):
    def __init__(self, cfg, dataset, sr, accelerator=None, metadata=None):
        super().__init__(cfg, dataset, sr, accelerator, metadata)

    def __getitem__(self, index):
        utt_info = self.metadata[index]
        wav_path = utt_info["Path"]

        # wav: (T,)
        wav = whisper.load_audio(wav_path, sr=16000)  # sr = 16000
        # convert to torch tensor
        wav = torch.from_numpy(wav)
        # record the length of wav without padding
        length = wav.shape[0]

        return utt_info, wav, length


def collate_batch(batch_list):
    """
    Args:
        batch_list: list of (metadata, wav, length)
    """
    metadata = [item[0] for item in batch_list]
    # wavs: (B, T)
    wavs = pad_sequence([item[1] for item in batch_list], batch_first=True)
    lens = [item[2] for item in batch_list]

    return metadata, wavs, lens
