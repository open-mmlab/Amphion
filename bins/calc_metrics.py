# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import numpy as np
import json
import argparse

from glob import glob
from tqdm import tqdm
from collections import defaultdict

from evaluation.metrics.energy.energy_rmse import extract_energy_rmse
from evaluation.metrics.energy.energy_pearson_coefficients import (
    extract_energy_pearson_coeffcients,
)
from evaluation.metrics.f0.f0_pearson_coefficients import extract_fpc
from evaluation.metrics.f0.f0_periodicity_rmse import extract_f0_periodicity_rmse
from evaluation.metrics.f0.f0_rmse import extract_f0rmse
from evaluation.metrics.f0.v_uv_f1 import extract_f1_v_uv
from evaluation.metrics.intelligibility.character_error_rate import extract_cer
from evaluation.metrics.intelligibility.word_error_rate import extract_wer
from evaluation.metrics.similarity.speaker_similarity import extract_speaker_similarity
from evaluation.metrics.similarity.resemblyzer_similarity import (
    extract_resemblyzer_similarity,
)
from evaluation.metrics.spectrogram.frechet_distance import extract_fad
from evaluation.metrics.spectrogram.mel_cepstral_distortion import extract_mcd
from evaluation.metrics.spectrogram.multi_resolution_stft_distance import extract_mstft
from evaluation.metrics.spectrogram.pesq import extract_pesq
from evaluation.metrics.spectrogram.scale_invariant_signal_to_distortion_ratio import (
    extract_si_sdr,
)
from evaluation.metrics.spectrogram.scale_invariant_signal_to_noise_ratio import (
    extract_si_snr,
)
from evaluation.metrics.spectrogram.short_time_objective_intelligibility import (
    extract_stoi,
)

METRIC_FUNC = {
    "energy_rmse": extract_energy_rmse,
    "energy_pc": extract_energy_pearson_coeffcients,
    "fpc": extract_fpc,
    "f0_periodicity_rmse": extract_f0_periodicity_rmse,
    "f0rmse": extract_f0rmse,
    "v_uv_f1": extract_f1_v_uv,
    "cer": extract_cer,
    "wer": extract_wer,
    "rawnet3_similarity": extract_speaker_similarity,
    "resemblyzer_similarity": extract_resemblyzer_similarity,
    "fad": extract_fad,
    "mcd": extract_mcd,
    "mstft": extract_mstft,
    "pesq": extract_pesq,
    "si_sdr": extract_si_sdr,
    "si_snr": extract_si_snr,
    "stoi": extract_stoi,
}


def calc_metric(ref_dir, deg_dir, dump_dir, metrics, fs=None):
    result = defaultdict()

    for metric in tqdm(metrics):
        if metric in ["fad", "rawnet3_similarity"]:
            result[metric] = str(METRIC_FUNC[metric](ref_dir, deg_dir))
            continue
        elif metric in ["resemblyzer_similarity"]:
            result[metric] = str(METRIC_FUNC[metric](deg_dir, ref_dir, dump_dir))
            continue

        audios_ref = []
        audios_deg = []

        files = glob(ref_dir + "/*.wav")

        for file in files:
            audios_ref.append(file)
            uid = file.split("/")[-1].split(".wav")[0]
            file_gt = deg_dir + "/{}.wav".format(uid)
            audios_deg.append(file_gt)

        if metric in ["v_uv_f1"]:
            tp_total = 0
            fp_total = 0
            fn_total = 0

            for i in tqdm(range(len(audios_ref))):
                audio_ref = audios_ref[i]
                audio_deg = audios_deg[i]
                tp, fp, fn = METRIC_FUNC[metric](audio_ref, audio_deg, fs)
                tp_total += tp
                fp_total += fp
                fn_total += fn

            result[metric] = str(tp_total / (tp_total + (fp_total + fn_total) / 2))
        else:
            scores = []

            for i in tqdm(range(len(audios_ref))):
                audio_ref = audios_ref[i]
                audio_deg = audios_deg[i]

                score = METRIC_FUNC[metric](
                    audio_ref=audio_ref, audio_deg=audio_deg, fs=fs
                )
                if not np.isnan(score):
                    scores.append(score)

            scores = np.array(scores)
            result["{}_mean".format(metric)] = str(np.mean(scores))
            result["{}_std".format(metric)] = str(np.std(scores))

    data = json.dumps(result, indent=4)

    with open(os.path.join(dump_dir, "result.json"), "w", newline="\n") as f:
        f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_dir",
        type=str,
        help="Path to the target audio folder.",
    )
    parser.add_argument(
        "--deg_dir",
        type=str,
        help="Path to the reference audio folder.",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        help="Path to dump the results.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics used to evaluate.",
    )
    parser.add_argument(
        "--fs",
        type=str,
        help="(Optional) Sampling rate",
    )
    args = parser.parse_args()

    calc_metric(args.ref_dir, args.deg_dir, args.dump_dir, args.metrics, args.fs)
