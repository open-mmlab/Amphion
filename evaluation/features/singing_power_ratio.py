# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import librosa

from utils.util import JsonHParams
from utils.f0 import get_f0_features_using_parselmouth, get_pitch_sub_median
from utils.mel import extract_mel_features


def extract_spr(
    audio,
    fs=None,
    hop_length=256,
    win_length=1024,
    n_fft=1024,
    n_mels=128,
    f0_min=37,
    f0_max=1000,
    pitch_bin=256,
    pitch_max=1100.0,
    pitch_min=50.0,
):
    """Compute Singing Power Ratio (SPR) from a given audio.
    audio: path to the audio.
    fs: sampling rate.
    hop_length: hop length.
    win_length: window length.
    n_mels: number of mel filters.
    f0_min: lower limit for f0.
    f0_max: upper limit for f0.
    pitch_bin: number of bins for f0 quantization.
    pitch_max: upper limit for f0 quantization.
    pitch_min: lower limit for f0 quantization.
    """
    # Load audio
    if fs != None:
        audio, _ = librosa.load(audio, sr=fs)
    else:
        audio, fs = librosa.load(audio)
    audio = torch.from_numpy(audio)

    # Initialize config
    cfg = JsonHParams()
    cfg.sample_rate = fs
    cfg.hop_size = hop_length
    cfg.win_size = win_length
    cfg.n_fft = n_fft
    cfg.n_mel = n_mels
    cfg.f0_min = f0_min
    cfg.f0_max = f0_max
    cfg.pitch_bin = pitch_bin
    cfg.pitch_max = pitch_max
    cfg.pitch_min = pitch_min

    # Extract mel spectrograms

    cfg.fmin = 2000
    cfg.fmax = 4000

    mel1 = extract_mel_features(
        y=audio.unsqueeze(0),
        cfg=cfg,
    ).squeeze(0)

    cfg.fmin = 0
    cfg.fmax = 2000

    mel2 = extract_mel_features(
        y=audio.unsqueeze(0),
        cfg=cfg,
    ).squeeze(0)

    f0 = get_f0_features_using_parselmouth(
        audio,
        cfg,
    )

    # Mel length alignment
    length = min(len(f0), mel1.shape[-1])
    f0 = f0[:length]
    mel1 = mel1[:, :length]
    mel2 = mel2[:, :length]

    # Compute SPR
    res = []

    for i in range(mel1.shape[-1]):
        if f0[i] <= 1:
            continue

        chunk1 = mel1[:, i]
        chunk2 = mel2[:, i]

        max1 = max(chunk1.numpy())
        max2 = max(chunk2.numpy())

        tmp_res = max2 - max1

        res.append(tmp_res)

    if len(res) == 0:
        return False
    else:
        return sum(res) / len(res)
