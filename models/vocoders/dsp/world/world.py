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
import json
import re
import torchaudio

from cuhkszsvc.configs.config_parse import get_wav_path, get_wav_file_path
from utils.io import has_existed


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


def extract_world_features(wave_file, fs, frameshift):
    # waveform: (1, seq)
    waveform, sample_rate = torchaudio.load(wave_file)
    if sample_rate != fs:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=fs
        )
    # x: (seq,)
    x = np.array(torch.clamp(waveform[0], -1.0, 1.0), dtype=np.double)

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


def extract_mcep_features_of_dataset(
    output_path, dataset_path, dataset, mcsize, fs, frameshift, splits=None
):
    output_dir = os.path.join(output_path, dataset, "mcep/{}".format(fs))

    if not splits:
        splits = ["train", "test"] if dataset != "m4singer" else ["test"]

    for dataset_type in splits:
        print("-" * 20)
        print("Dataset: {}, {}".format(dataset, dataset_type))

        output_file = os.path.join(output_dir, "{}.pkl".format(dataset_type))
        if has_existed(output_file):
            continue

        # Extract SP features
        print("\nExtracting SP featuers...")
        sp_features = get_world_features_of_dataset(
            output_path, dataset_path, dataset, dataset_type, fs, frameshift
        )

        # SP to MCEP
        print("\nTransform SP to MCEP...")
        mcep_features = [sp2mcep(sp, mcsize=mcsize, fs=fs) for sp in tqdm(sp_features)]

        # Save
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(mcep_features, f)


def get_world_features_of_dataset(
    output_path,
    dataset_path,
    dataset,
    dataset_type,
    fs,
    frameshift,
    save_sp_feature=False,
):
    data_dir = os.path.join(output_path, dataset)
    wave_dir = get_wav_path(dataset_path, dataset)

    # Dataset
    dataset_file = os.path.join(data_dir, "{}.json".format(dataset_type))
    if not os.path.exists(dataset_file):
        print("File {} has not existed.".format(dataset_file))
        return None

    with open(dataset_file, "r") as f:
        datasets = json.load(f)

    # Save dir
    f0_dir = os.path.join(output_path, dataset, "f0")
    os.makedirs(f0_dir, exist_ok=True)

    # Extract
    f0_features = []
    sp_features = []
    for utt in tqdm(datasets):
        wave_file = get_wav_file_path(dataset, wave_dir, utt)
        f0, sp, _, _ = extract_world_features(wave_file, fs, frameshift)

        sp_features.append(sp)
        f0_features.append(f0)

    # Save sp
    if save_sp_feature:
        sp_dir = os.path.join(output_path, dataset, "sp")
        os.makedirs(sp_dir, exist_ok=True)
        with open(os.path.join(sp_dir, "{}.pkl".format(dataset_type)), "wb") as f:
            pickle.dump(sp_features, f)

    # F0 statistics
    f0_statistics_file = os.path.join(f0_dir, "{}_f0.pkl".format(dataset_type))
    f0_statistics(f0_features, f0_statistics_file)

    return sp_features


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
