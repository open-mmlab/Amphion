# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import librosa
import torch

import numpy as np

from utils.util import JsonHParams
from utils.f0 import get_f0_features_using_parselmouth


ZERO = 1e-8


def extract_f1_v_uv(
    audio_ref,
    audio_deg,
    hop_length=256,
    f0_min=50,
    f0_max=1100,
    **kwargs,
):
    """Compute F1 socre of voiced/unvoiced accuracy between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    hop_length: hop length.
    f0_min: lower limit for f0.
    f0_max: upper limit for f0.
    pitch_bin: number of bins for f0 quantization.
    pitch_max: upper limit for f0 quantization.
    pitch_min: lower limit for f0 quantization.
    need_mean: subtract the mean value from f0 if "True".
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

    # Initialize config
    cfg = JsonHParams()
    cfg.sample_rate = fs
    cfg.hop_size = hop_length
    cfg.f0_min = f0_min
    cfg.f0_max = f0_max
    cfg.pitch_bin = 256
    cfg.pitch_max = f0_max
    cfg.pitch_min = f0_min

    # Compute f0
    f0_ref = get_f0_features_using_parselmouth(
        audio_ref,
        cfg,
    )

    f0_deg = get_f0_features_using_parselmouth(
        audio_deg,
        cfg,
    )

    # Avoid silence
    min_length = min(len(f0_ref), len(f0_deg))
    if min_length <= 1:
        return 0, 0, 0

    # F0 length alignment
    if method == "cut":
        length = min(len(f0_ref), len(f0_deg))
        f0_ref = f0_ref[:length]
        f0_deg = f0_deg[:length]
    elif method == "dtw":
        _, wp = librosa.sequence.dtw(f0_ref, f0_deg, backtrack=True)
        f0_gt_new = []
        f0_pred_new = []
        for i in range(wp.shape[0]):
            gt_index = wp[i][0]
            pred_index = wp[i][1]
            f0_gt_new.append(f0_ref[gt_index])
            f0_pred_new.append(f0_deg[pred_index])
        f0_ref = np.array(f0_gt_new)
        f0_deg = np.array(f0_pred_new)
        assert len(f0_ref) == len(f0_deg)

    # Get voiced/unvoiced parts
    ref_voiced = torch.Tensor([f0_ref != 0]).bool()
    deg_voiced = torch.Tensor([f0_deg != 0]).bool()

    # Compute TP, FP, FN
    true_postives = (ref_voiced & deg_voiced).sum()
    false_postives = (~ref_voiced & deg_voiced).sum()
    false_negatives = (ref_voiced & ~deg_voiced).sum()

    return (
        true_postives.detach().cpu().numpy().tolist(),
        false_postives.detach().cpu().numpy().tolist(),
        false_negatives.detach().cpu().numpy().tolist(),
    )
