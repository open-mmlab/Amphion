# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import librosa

import numpy as np

from pypesq import pesq


def extract_pesq(audio_ref, audio_deg, **kwargs):
    """Extract PESQ for a two given audio.
    audio1: the given reference audio. It is a numpy array.
    audio2: the given synthesized audio. It is a numpy array.
    fs: sampling rate.
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    """
    # Load hyperparameters
    kwargs = kwargs["kwargs"]
    fs = kwargs["fs"]
    method = kwargs["method"]

    # Load audio
    if fs != None:
        audio_ref, _ = librosa.load(audio_ref, sr=fs)
        audio_deg, _ = librosa.load(audio_deg, sr=fs)
    else:
        audio_ref, fs = librosa.load(audio_ref)
        audio_deg, fs = librosa.load(audio_deg)

    # Resample
    if fs != 16000:
        audio_ref = librosa.resample(audio_ref, orig_sr=fs, target_sr=16000)
        audio_deg = librosa.resample(audio_deg, orig_sr=fs, target_sr=16000)
        fs = 16000

    # Audio length alignment
    if len(audio_ref) != len(audio_deg):
        if method == "cut":
            length = min(len(audio_ref), len(audio_deg))
            audio_ref = audio_ref[:length]
            audio_deg = audio_deg[:length]
        elif method == "dtw":
            _, wp = librosa.sequence.dtw(audio_ref, audio_deg, backtrack=True)
            audio_ref_new = []
            audio_deg_new = []
            for i in range(wp.shape[0]):
                ref_index = wp[i][0]
                deg_index = wp[i][1]
                audio_ref_new.append(audio_ref[ref_index])
                audio_deg_new.append(audio_deg[deg_index])
            audio_ref = np.array(audio_ref_new)
            audio_deg = np.array(audio_deg_new)
            assert len(audio_ref) == len(audio_deg)

    # Compute pesq
    score = pesq(audio_ref, audio_deg, fs)
    return score
