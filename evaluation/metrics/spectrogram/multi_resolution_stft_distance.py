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
    fs=None,
    mid_freq=None,
    high_freq=None,
    method="cut",
    version="pwg",
):
    """Compute Multi-Scale STFT Distance (mstft) between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    med_freq: division frequency for mid frequency parts.
    high_freq: division frequency for high frequency parts.
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    version: "pwg" will use the computational version provided by ParallelWaveGAN.
             "encodec" will use the computational version provided by Encodec.
    """
    # Load audio
    if fs != None:
        audio_ref, _ = librosa.load(audio_ref, sr=fs)
        audio_deg, _ = librosa.load(audio_deg, sr=fs)
    else:
        audio_ref, fs = librosa.load(audio_ref)
        audio_deg, fs = librosa.load(audio_deg)

    # Automatically choose mid_freq and high_freq if they are not given
    if mid_freq == None:
        mid_freq = fs // 6
    if high_freq == None:
        high_freq = fs // 3

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
    l2Loss = torch.nn.MSELoss(reduction="mean")

    # Compute distance
    if version == "encodec":
        n_fft = 1024

        mstft = 0
        mstft_low = 0
        mstft_mid = 0
        mstft_high = 0

        freq_resolution = fs / n_fft
        mid_freq_index = 1 + int(np.floor(mid_freq / freq_resolution))
        high_freq_index = 1 + int(np.floor(high_freq / freq_resolution))

        for i in range(5, 11):
            hop_length = 2**i // 4
            win_length = 2**i

            spec_ref = librosa.stft(
                y=audio_ref, n_fft=n_fft, hop_length=hop_length, win_length=win_length
            )
            spec_deg = librosa.stft(
                y=audio_deg, n_fft=n_fft, hop_length=hop_length, win_length=win_length
            )

            mag_ref = np.abs(spec_ref)
            mag_deg = np.abs(spec_deg)

            mag_ref = torch.from_numpy(mag_ref)
            mag_deg = torch.from_numpy(mag_deg)
            mstft += l1Loss(mag_ref, mag_deg) + l2Loss(mag_ref, mag_deg)

            mag_ref_low = mag_ref[:mid_freq_index, :]
            mag_deg_low = mag_deg[:mid_freq_index, :]
            mstft_low += l1Loss(mag_ref_low, mag_deg_low) + l2Loss(
                mag_ref_low, mag_deg_low
            )

            mag_ref_mid = mag_ref[mid_freq_index:high_freq_index, :]
            mag_deg_mid = mag_deg[mid_freq_index:high_freq_index, :]
            mstft_mid += l1Loss(mag_ref_mid, mag_deg_mid) + l2Loss(
                mag_ref_mid, mag_deg_mid
            )

            mag_ref_high = mag_ref[high_freq_index:, :]
            mag_deg_high = mag_deg[high_freq_index:, :]
            mstft_high += l1Loss(mag_ref_high, mag_deg_high) + l2Loss(
                mag_ref_high, mag_deg_high
            )

        mstft /= 6
        mstft_low /= 6
        mstft_mid /= 6
        mstft_high /= 6

        return mstft
    elif version == "pwg":
        fft_sizes = [1024, 2048, 512]
        hop_sizes = [120, 240, 50]
        win_sizes = [600, 1200, 240]

        audio_ref = torch.from_numpy(audio_ref)
        audio_deg = torch.from_numpy(audio_deg)

        mstft_sc = 0
        mstft_sc_low = 0
        mstft_sc_mid = 0
        mstft_sc_high = 0

        mstft_mag = 0
        mstft_mag_low = 0
        mstft_mag_mid = 0
        mstft_mag_high = 0

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
            sc_loss = torch.norm(mag_ref - mag_deg, p="fro") / torch.norm(
                mag_ref, p="fro"
            )
            mag_loss = l1Loss(torch.log(mag_ref), torch.log(mag_deg))

            mstft_sc += sc_loss
            mstft_mag += mag_loss

            freq_resolution = fs / n_fft
            mid_freq_index = 1 + int(np.floor(mid_freq / freq_resolution))
            high_freq_index = 1 + int(np.floor(high_freq / freq_resolution))

            mag_ref_low = mag_ref[:, :mid_freq_index]
            mag_deg_low = mag_deg[:, :mid_freq_index]
            sc_loss_low = torch.norm(mag_ref_low - mag_deg_low, p="fro") / torch.norm(
                mag_ref_low, p="fro"
            )
            mag_loss_low = l1Loss(torch.log(mag_ref_low), torch.log(mag_deg_low))

            mstft_sc_low += sc_loss_low
            mstft_mag_low += mag_loss_low

            mag_ref_mid = mag_ref[:, mid_freq_index:high_freq_index]
            mag_deg_mid = mag_deg[:, mid_freq_index:high_freq_index]
            sc_loss_mid = torch.norm(mag_ref_mid - mag_deg_mid, p="fro") / torch.norm(
                mag_ref_mid, p="fro"
            )
            mag_loss_mid = l1Loss(torch.log(mag_ref_mid), torch.log(mag_deg_mid))

            mstft_sc_mid += sc_loss_mid
            mstft_mag_mid += mag_loss_mid

            mag_ref_high = mag_ref[:, high_freq_index:]
            mag_deg_high = mag_deg[:, high_freq_index:]
            sc_loss_high = torch.norm(
                mag_ref_high - mag_deg_high, p="fro"
            ) / torch.norm(mag_ref_high, p="fro")
            mag_loss_high = l1Loss(torch.log(mag_ref_high), torch.log(mag_deg_high))

            mstft_sc_high += sc_loss_high
            mstft_mag_high += mag_loss_high

        # Normalize distances
        mstft_sc /= len(fft_sizes)
        mstft_sc_low /= len(fft_sizes)
        mstft_sc_mid /= len(fft_sizes)
        mstft_sc_high /= len(fft_sizes)

        mstft_mag /= len(fft_sizes)
        mstft_mag_low /= len(fft_sizes)
        mstft_mag_mid /= len(fft_sizes)
        mstft_mag_high /= len(fft_sizes)

        # return (
        #     mstft_sc.numpy().tolist(),
        #     mstft_sc_low.numpy().tolist(),
        #     mstft_sc_mid.numpy().tolist(),
        #     mstft_sc_high.numpy().tolist(),
        #     mstft_mag.numpy().tolist(),
        #     mstft_mag_low.numpy().tolist(),
        #     mstft_mag_mid.numpy().tolist(),
        #     mstft_mag_high.numpy().tolist(),
        # )

        return mstft_sc.numpy().tolist() + mstft_mag.numpy().tolist()
