# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import numpy as np
import json
import argparse
import whisper
import torch

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
from evaluation.metrics.similarity.speaker_similarity import extract_similarity
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
    "similarity": extract_similarity,
    "fad": extract_fad,
    "mcd": extract_mcd,
    "mstft": extract_mstft,
    "pesq": extract_pesq,
    "si_sdr": extract_si_sdr,
    "si_snr": extract_si_snr,
    "stoi": extract_stoi,
}


def calc_metric(
    ref_dir,
    deg_dir,
    dump_dir,
    metrics,
    **kwargs,
):
    result = defaultdict()

    for metric in tqdm(metrics):
        if metric in ["fad", "similarity"]:
            result[metric] = str(METRIC_FUNC[metric](ref_dir, deg_dir, kwargs=kwargs))
            continue

        audios_ref = []
        audios_deg = []

        files = glob(deg_dir + "/*.wav")

        for file in files:
            audios_deg.append(file)
            uid = file.split("/")[-1].split(".wav")[0]
            file_gt = ref_dir + "/{}.wav".format(uid)
            audios_ref.append(file_gt)

        if metric in ["wer", "cer"] and kwargs["intelligibility_mode"] == "gt_content":
            ltr_path = kwargs["ltr_path"]
            tmpltrs = {}
            with open(ltr_path, "r") as f:
                for line in f:
                    paras = line.replace("\n", "").split("|")
                    paras[1] = paras[1].replace(" ", "")
                    paras[1] = paras[1].replace(".", "")
                    paras[1] = paras[1].replace("'", "")
                    paras[1] = paras[1].replace("-", "")
                    paras[1] = paras[1].replace(",", "")
                    paras[1] = paras[1].replace("!", "")
                    paras[1] = paras[1].lower()
                    tmpltrs[paras[0]] = paras[1]
            ltrs = []
            files = glob(ref_dir + "/*.wav")
            for file in files:
                ltrs.append(tmpltrs[os.path.basename(file)])

        if metric in ["v_uv_f1"]:
            tp_total = 0
            fp_total = 0
            fn_total = 0

            for i in tqdm(range(len(audios_ref))):
                audio_ref = audios_ref[i]
                audio_deg = audios_deg[i]
                tp, fp, fn = METRIC_FUNC[metric](audio_ref, audio_deg, kwargs=kwargs)
                tp_total += tp
                fp_total += fp
                fn_total += fn

            result[metric] = str(tp_total / (tp_total + (fp_total + fn_total) / 2))
        else:
            scores = []
            for i in tqdm(range(len(audios_ref))):
                audio_ref = audios_ref[i]
                audio_deg = audios_deg[i]

                if metric in ["wer", "cer"]:
                    model = whisper.load_model("large")
                    mode = kwargs["intelligibility_mode"]
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        model = model.to(device)

                    if mode == "gt_audio":
                        kwargs["audio_ref"] = audio_ref
                        kwargs["audio_deg"] = audio_deg
                        score = METRIC_FUNC[metric](
                            model,
                            kwargs=kwargs,
                        )
                    elif mode == "gt_content":
                        kwargs["content_gt"] = ltrs[i]
                        kwargs["audio_deg"] = audio_deg
                        score = METRIC_FUNC[metric](
                            model,
                            kwargs=kwargs,
                        )
                else:
                    score = METRIC_FUNC[metric](
                        audio_ref,
                        audio_deg,
                        kwargs=kwargs,
                    )
                if not np.isnan(score):
                    scores.append(score)

            scores = np.array(scores)
            result["{}".format(metric)] = str(np.mean(scores))

    data = json.dumps(result, indent=4)

    with open(os.path.join(dump_dir, "result.json"), "w", newline="\n") as f:
        f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_dir",
        type=str,
        help="Path to the reference audio folder.",
    )
    parser.add_argument(
        "--deg_dir",
        type=str,
        help="Path to the test audio folder.",
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
        default="None",
        help="(Optional) Sampling rate",
    )
    parser.add_argument(
        "--align_method",
        type=str,
        default="dtw",
        help="(Optional) Method for aligning feature length. ['cut', 'dtw']",
    )

    parser.add_argument(
        "--db_scale",
        type=str,
        default="True",
        help="(Optional) Wether or not computing energy related metrics in db scale.",
    )
    parser.add_argument(
        "--f0_subtract_mean",
        type=str,
        default="True",
        help="(Optional) Wether or not computing f0 related metrics with mean value subtracted.",
    )

    parser.add_argument(
        "--similarity_model",
        type=str,
        default="wavlm",
        help="(Optional)The model for computing speaker similarity. ['rawnet', 'wavlm', 'resemblyzer']",
    )
    parser.add_argument(
        "--similarity_mode",
        type=str,
        default="pairwith",
        help="(Optional)The method of calculating similarity, where set to overall means computing \
        the speaker similarity between two folder of audios content freely, and set to pairwith means \
        computing the speaker similarity between a seires of paired gt/pred audios",
    )

    parser.add_argument(
        "--ltr_path",
        type=str,
        default="None",
        help="(Optional)Path to the transcription file,Note that the format in the transcription \
            file is 'file name|transcription'",
    )
    parser.add_argument(
        "--intelligibility_mode",
        type=str,
        default="gt_audio",
        help="(Optional)The method of calculating WER and CER, where set to gt_audio means selecting \
        the recognition content of the reference audio as the target, and set to gt_content means \
        using transcription as the target",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="(Optional)['english','chinese']",
    )

    args = parser.parse_args()

    calc_metric(
        args.ref_dir,
        args.deg_dir,
        args.dump_dir,
        args.metrics,
        fs=int(args.fs) if args.fs != "None" else None,
        method=args.align_method,
        db_scale=True if args.db_scale == "True" else False,
        need_mean=True if args.f0_subtract_mean == "True" else False,
        model_name=args.similarity_model,
        similarity_mode=args.similarity_mode,
        ltr_path=args.ltr_path,
        intelligibility_mode=args.intelligibility_mode,
        language=args.language,
    )
