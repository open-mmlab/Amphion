# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 1. Extract WORLD features including F0, AP, SP
# 2. Transform between SP and MCEP
import torchaudio
import pyworld as pw
import numpy as np
import torch
import diffsptk
import os
from tqdm import tqdm
import pickle
import torchaudio


def get_mcep_params(fs):
    """Hyperparameters of transformation between SP and MCEP

    Reference:
        https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/world_v2/copy_synthesis.sh

    """
    if fs in [44100, 48000]:
        fft_size = 2048
        alpha = 0.77
    if fs in [16000]:
        fft_size = 1024
        alpha = 0.58
    return fft_size, alpha


def extract_world_features(waveform, frameshift=10):
    # waveform: (1, seq)
    # x: (seq,)
    x = np.array(waveform, dtype=np.double)

    _f0, t = pw.dio(x, fs, frame_period=frameshift)  # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, fs)  # extract aperiodicity

    return f0, sp, ap, fs


def sp2mcep(x, mcsize, fs):
    fft_size, alpha = get_mcep_params(fs)
    x = torch.as_tensor(x, dtype=torch.float)

    tmp = diffsptk.ScalarOperation("SquareRoot")(x)
    tmp = diffsptk.ScalarOperation("Multiplication", 32768.0)(tmp)
    mgc = diffsptk.MelCepstralAnalysis(
        cep_order=mcsize - 1, fft_length=fft_size, alpha=alpha, n_iter=1
    )(tmp)
    return mgc.numpy()


def mcep2sp(x, mcsize, fs):
    fft_size, alpha = get_mcep_params(fs)
    x = torch.as_tensor(x, dtype=torch.float)

    tmp = diffsptk.MelGeneralizedCepstrumToSpectrum(
        alpha=alpha,
        cep_order=mcsize - 1,
        fft_length=fft_size,
    )(x)
    tmp = diffsptk.ScalarOperation("Division", 32768.0)(tmp)
    sp = diffsptk.ScalarOperation("Power", 2)(tmp)
    return sp.double().numpy()


def f0_statistics(f0_features, path):
    print("\nF0 statistics...")

    total_f0 = []
    for f0 in tqdm(f0_features):
        total_f0 += [f for f in f0 if f != 0]

    mean = sum(total_f0) / len(total_f0)
    print("Min = {}, Max = {}, Mean = {}".format(min(total_f0), max(total_f0), mean))

    with open(path, "wb") as f:
        pickle.dump([mean, total_f0], f)


def world_synthesis(f0, sp, ap, fs, frameshift):
    y = pw.synthesize(
        f0, sp, ap, fs, frame_period=frameshift
    )  # synthesize an utterance using the parameters
    return y
