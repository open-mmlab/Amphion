# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import librosa
from scipy import signal


def extract_ltas(audio, fs=None, n_fft=1024, hop_length=256):
    """Extract Long-Term Average Spectrum for a given audio."""
    if fs != None:
        y, _ = librosa.load(audio, sr=fs)
    else:
        y, fs = librosa.load(audio)
    frequency, density = signal.welch(
        x=y, fs=fs, window="hann", nperseg=hop_length, nfft=n_fft
    )
    return frequency, density
