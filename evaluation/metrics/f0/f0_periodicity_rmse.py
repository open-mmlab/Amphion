# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchcrepe
import math
import librosa
import torch

import numpy as np


def extract_f0_periodicity_rmse(
    audio_ref,
    audio_deg,
    hop_length=256,
    **kwargs,
):
    """Compute f0 periodicity Root Mean Square Error (RMSE) between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    hop_length: hop length.
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

    # Convert to torch
    audio_ref = torch.from_numpy(audio_ref).unsqueeze(0)
    audio_deg = torch.from_numpy(audio_deg).unsqueeze(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Get periodicity
    _, periodicity_ref = torchcrepe.predict(
        audio_ref,
        sample_rate=fs,
        hop_length=hop_length,
        fmin=0,
        fmax=1500,
        model="full",
        return_periodicity=True,
        device=device,
    )
    _, periodicity_deg = torchcrepe.predict(
        audio_deg,
        sample_rate=fs,
        hop_length=hop_length,
        fmin=0,
        fmax=1500,
        model="full",
        return_periodicity=True,
        device=device,
    )

    # Cut silence
    periodicity_ref = (
        torchcrepe.threshold.Silence()(
            periodicity_ref,
            audio_ref,
            fs,
            hop_length=hop_length,
        )
        .squeeze(0)
        .numpy()
    )
    periodicity_deg = (
        torchcrepe.threshold.Silence()(
            periodicity_deg,
            audio_deg,
            fs,
            hop_length=hop_length,
        )
        .squeeze(0)
        .numpy()
    )

    # Avoid silence audio
    min_length = min(len(periodicity_ref), len(periodicity_deg))
    if min_length <= 1:
        return 0

    # Periodicity length alignment
    if method == "cut":
        length = min(len(periodicity_ref), len(periodicity_deg))
        periodicity_ref = periodicity_ref[:length]
        periodicity_deg = periodicity_deg[:length]
    elif method == "dtw":
        _, wp = librosa.sequence.dtw(periodicity_ref, periodicity_deg, backtrack=True)
        periodicity_ref_new = []
        periodicity_deg_new = []
        for i in range(wp.shape[0]):
            ref_index = wp[i][0]
            deg_index = wp[i][1]
            periodicity_ref_new.append(periodicity_ref[ref_index])
            periodicity_deg_new.append(periodicity_deg[deg_index])
        periodicity_ref = np.array(periodicity_ref_new)
        periodicity_deg = np.array(periodicity_deg_new)
        assert len(periodicity_ref) == len(periodicity_deg)

    # Compute RMSE
    periodicity_mse = np.square(np.subtract(periodicity_ref, periodicity_deg)).mean()
    periodicity_rmse = math.sqrt(periodicity_mse)

    return periodicity_rmse
