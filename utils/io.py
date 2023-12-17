# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import torchaudio


def save_feature(process_dir, feature_dir, item, feature, overrides=True):
    """Save features to path

    Args:
        process_dir (str): directory to store features
        feature_dir (_type_): directory to store one type of features (mel, energy, ...)
        item (str): uid
        feature (tensor): feature tensor
        overrides (bool, optional): whether to override existing files. Defaults to True.
    """
    process_dir = os.path.join(process_dir, feature_dir)
    os.makedirs(process_dir, exist_ok=True)
    out_path = os.path.join(process_dir, item + ".npy")

    if os.path.exists(out_path):
        if overrides:
            np.save(out_path, feature)
    else:
        np.save(out_path, feature)


def save_txt(process_dir, feature_dir, item, feature, overrides=True):
    process_dir = os.path.join(process_dir, feature_dir)
    os.makedirs(process_dir, exist_ok=True)
    out_path = os.path.join(process_dir, item + ".txt")

    if os.path.exists(out_path):
        if overrides:
            f = open(out_path, "w")
            f.writelines(feature)
            f.close()
    else:
        f = open(out_path, "w")
        f.writelines(feature)
        f.close()


def save_audio(path, waveform, fs, add_silence=False, turn_up=False, volume_peak=0.9):
    """Save audio to path with processing  (turn up volume, add silence)
    Args:
        path (str): path to save audio
        waveform (numpy array): waveform to save
        fs (int): sampling rate
        add_silence (bool, optional): whether to add silence to beginning and end. Defaults to False.
        turn_up (bool, optional): whether to turn up volume. Defaults to False.
        volume_peak (float, optional): volume peak. Defaults to 0.9.
    """
    if turn_up:
        # continue to turn up to volume_peak
        ratio = volume_peak / max(waveform.max(), abs(waveform.min()))
        waveform = waveform * ratio

    if add_silence:
        silence_len = fs // 20
        silence = np.zeros((silence_len,), dtype=waveform.dtype)
        result = np.concatenate([silence, waveform, silence])
        waveform = result

    waveform = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
    if len(waveform.size()) == 1:
        waveform = waveform[None, :]
    elif waveform.size(0) != 1:
        # Stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    torchaudio.save(path, waveform, fs, encoding="PCM_S", bits_per_sample=16)


def save_torch_audio(process_dir, feature_dir, item, wav_torch, fs, overrides=True):
    """Save torch audio to path without processing
    Args:
        process_dir (str): directory to store features
        feature_dir (_type_): directory to store one type of features (mel, energy, ...)
        item (str): uid
        wav_torch (tensor): feature tensor
        fs (int): sampling rate
        overrides (bool, optional): whether to override existing files. Defaults to True.
    """
    if wav_torch.shape != 2:
        wav_torch = wav_torch.unsqueeze(0)

    process_dir = os.path.join(process_dir, feature_dir)
    os.makedirs(process_dir, exist_ok=True)
    out_path = os.path.join(process_dir, item + ".wav")

    torchaudio.save(out_path, wav_torch, fs)


async def async_load_audio(path, sample_rate: int = 24000):
    r"""
    Args:
        path: The source loading path.
        sample_rate: The target sample rate, will automatically resample if necessary.

    Returns:
        waveform: The waveform object. Should be [1 x sequence_len].
    """

    async def use_torchaudio_load(path):
        return torchaudio.load(path)

    waveform, sr = await use_torchaudio_load(path)
    waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    if torch.any(torch.isnan(waveform) or torch.isinf(waveform)):
        raise ValueError("NaN or Inf found in waveform.")
    return waveform


async def async_save_audio(
    path,
    waveform,
    sample_rate: int = 24000,
    add_silence: bool = False,
    volume_peak: float = 0.9,
):
    r"""
    Args:
        path: The target saving path.
        waveform: The waveform object. Should be [n_channel x sequence_len].
        sample_rate: Sample rate.
        add_silence: If ``true``, concat 0.05s silence to beginning and end.
        volume_peak: Turn up volume for larger number, vice versa.
    """

    async def use_torchaudio_save(path, waveform, sample_rate):
        torchaudio.save(
            path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16
        )

    waveform = torch.as_tensor(waveform, device="cpu", dtype=torch.float32)
    shape = waveform.size()[:-1]

    ratio = abs(volume_peak) / max(waveform.max(), abs(waveform.min()))
    waveform = waveform * ratio

    if add_silence:
        silence_len = sample_rate // 20
        silence = torch.zeros((*shape, silence_len), dtype=waveform.type())
        waveform = torch.concatenate((silence, waveform, silence), dim=-1)

    if waveform.dim() == 1:
        waveform = waveform[None]

    await use_torchaudio_save(path, waveform, sample_rate)


def load_mel_extrema(cfg, dataset_name, split):
    dataset_dir = os.path.join(
        cfg.OUTPUT_PATH,
        "preprocess/{}_version".format(cfg.data.process_version),
        dataset_name,
    )

    min_file = os.path.join(
        dataset_dir,
        "mel_min_max",
        split.split("_")[-1],
        "mel_min.npy",
    )
    max_file = os.path.join(
        dataset_dir,
        "mel_min_max",
        split.split("_")[-1],
        "mel_max.npy",
    )
    mel_min = np.load(min_file)
    mel_max = np.load(max_file)
    return mel_min, mel_max
