# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import librosa
import torch

import numpy as np
from numpy import linalg as LA

from torchmetrics import PearsonCorrCoef


def extract_energy_pearson_coeffcients(
    audio_ref,
    audio_deg,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    **kwargs,
):
    """Compute Energy Pearson Coefficients between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    n_fft: fft size.
    hop_length: hop length.
    win_length: window length.
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    db_scale: the ground truth and predicted audio will be converted to db_scale if "True".
    """
    # Load hyperparameters
    kwargs = kwargs["kwargs"]
    fs = kwargs["fs"]
    method = kwargs["method"]
    db_scale = kwargs["db_scale"]

    # Initialize method
    pearson = PearsonCorrCoef()

    # Load audio
    if fs != None:
        audio_ref, _ = librosa.load(audio_ref, sr=fs)
        audio_deg, _ = librosa.load(audio_deg, sr=fs)
    else:
        audio_ref, fs = librosa.load(audio_ref)
        audio_deg, fs = librosa.load(audio_deg)

    # STFT
    spec_ref = librosa.stft(
        y=audio_ref, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    spec_deg = librosa.stft(
        y=audio_deg, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )

    # Get magnitudes
    mag_ref = np.abs(spec_ref).T
    mag_deg = np.abs(spec_deg).T

    # Convert spectrogram to energy
    energy_ref = LA.norm(mag_ref, axis=1)
    energy_deg = LA.norm(mag_deg, axis=1)

    # Convert to db_scale
    if db_scale:
        energy_ref = 20 * np.log10(energy_ref)
        energy_deg = 20 * np.log10(energy_deg)

    # Audio length alignment
    if method == "cut":
        length = min(len(energy_ref), len(energy_deg))
        energy_ref = energy_ref[:length]
        energy_deg = energy_deg[:length]
    elif method == "dtw":
        _, wp = librosa.sequence.dtw(energy_ref, energy_deg, backtrack=True)
        energy_gt_new = []
        energy_pred_new = []
        for i in range(wp.shape[0]):
            gt_index = wp[i][0]
            pred_index = wp[i][1]
            energy_gt_new.append(energy_ref[gt_index])
            energy_pred_new.append(energy_deg[pred_index])
        energy_ref = np.array(energy_gt_new)
        energy_deg = np.array(energy_pred_new)
        assert len(energy_ref) == len(energy_deg)

    # Convert to tensor
    energy_ref = torch.from_numpy(energy_ref)
    energy_deg = torch.from_numpy(energy_deg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        energy_ref = energy_ref.to(device)
        energy_deg = energy_deg.to(device)
        pearson = pearson.to(device)

    return pearson(energy_ref, energy_deg).detach().cpu().numpy().tolist()
