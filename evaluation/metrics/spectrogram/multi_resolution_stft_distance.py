# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import librosa
import torch

import numpy as np


def extract_mstft(
    audio_ref,
    audio_deg,
    **kwargs,
):
    """Compute Multi-Scale STFT Distance (mstft) between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    med_freq: division frequency for mid frequency parts.
    high_freq: division frequency for high frequency parts.
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

    # Define loss function
    l1Loss = torch.nn.L1Loss(reduction="mean")

    # Compute distance
    fft_sizes = [1024, 2048, 512]
    hop_sizes = [120, 240, 50]
    win_sizes = [600, 1200, 240]

    audio_ref = torch.from_numpy(audio_ref)
    audio_deg = torch.from_numpy(audio_deg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        audio_ref = audio_ref.to(device)
        audio_deg = audio_deg.to(device)

    mstft_sc = 0
    mstft_mag = 0

    for n_fft, hop_length, win_length in zip(fft_sizes, hop_sizes, win_sizes):
        spec_ref = torch.stft(
            audio_ref, n_fft, hop_length, win_length, return_complex=False
        )
        spec_deg = torch.stft(
            audio_deg, n_fft, hop_length, win_length, return_complex=False
        )

        real_ref = spec_ref[..., 0]
        imag_ref = spec_ref[..., 1]
        real_deg = spec_deg[..., 0]
        imag_deg = spec_deg[..., 1]

        mag_ref = torch.sqrt(
            torch.clamp(real_ref**2 + imag_ref**2, min=1e-7)
        ).transpose(1, 0)
        mag_deg = torch.sqrt(
            torch.clamp(real_deg**2 + imag_deg**2, min=1e-7)
        ).transpose(1, 0)
        sc_loss = torch.norm(mag_ref - mag_deg, p="fro") / torch.norm(mag_ref, p="fro")
        mag_loss = l1Loss(torch.log(mag_ref), torch.log(mag_deg))

        mstft_sc += sc_loss
        mstft_mag += mag_loss

    # Normalize distances
    mstft_sc /= len(fft_sizes)
    mstft_mag /= len(fft_sizes)

    return (
        mstft_sc.detach().cpu().numpy().tolist()
        + mstft_mag.detach().cpu().numpy().tolist()
    )
