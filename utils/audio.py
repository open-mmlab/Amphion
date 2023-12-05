# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from numpy import linalg as LA
import librosa
import soundfile as sf
import librosa.filters


def load_audio_torch(wave_file, fs):
    """Load audio data into torch tensor

    Args:
        wave_file (str): path to wave file
        fs (int): sample rate

    Returns:
        audio (tensor): audio data in tensor
        fs (int): sample rate
    """

    audio, sample_rate = librosa.load(wave_file, sr=fs, mono=True)
    # audio: (T,)
    assert len(audio) > 2

    # Check the audio type (for soundfile loading backbone) - float, 8bit or 16bit
    if np.issubdtype(audio.dtype, np.integer):
        max_mag = -np.iinfo(audio.dtype).min
    else:
        max_mag = max(np.amax(audio), -np.amin(audio))
        max_mag = (
            (2**31) + 1
            if max_mag > (2**15)
            else ((2**15) + 1 if max_mag > 1.01 else 1.0)
        )

    # Normalize the audio
    audio = torch.FloatTensor(audio.astype(np.float32)) / max_mag

    if (torch.isnan(audio) | torch.isinf(audio)).any():
        return [], sample_rate or fs or 48000

    # Resample the audio to our target samplerate
    if fs is not None and fs != sample_rate:
        audio = torch.from_numpy(
            librosa.core.resample(audio.numpy(), orig_sr=sample_rate, target_sr=fs)
        )
        sample_rate = fs

    return audio, fs


def _stft(y, cfg):
    return librosa.stft(
        y=y, n_fft=cfg.n_fft, hop_length=cfg.hop_size, win_length=cfg.win_size
    )


def energy(wav, cfg):
    D = _stft(wav, cfg)
    magnitudes = np.abs(D).T  # [F, T]
    return LA.norm(magnitudes, axis=1)


def get_energy_from_tacotron(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    mel, energy = _stft.mel_spectrogram(audio)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return mel, energy
