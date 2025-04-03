# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from huggingface_hub import snapshot_download

from models.svc.vevosing.vevosing_utils import *


def vevosing_fm(content_wav_path, reference_wav_path, output_path):
    gen_audio = inference_pipeline.inference_fm(
        src_wav_path=content_wav_path,
        timbre_ref_wav_path=reference_wav_path,
        flow_matching_steps=32,
    )
    save_audio(gen_audio, output_path=output_path)


if __name__ == "__main__":
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/VevoSing",
        repo_type="model",
        cache_dir="./ckpts/VevoSing",
        allow_patterns=["tokenizer/contentstyle_fvq16384_12.5hz/*"],
    )
    tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
    )

    # ===== Flow Matching Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/VevoSing",
        repo_type="model",
        cache_dir="./ckpts/VevoSing",
        allow_patterns=["acoustic_modeling/contentstyle_fvq16384_12.5hz_to_24kmels/*"],
    )

    fmt_cfg_path = (
        "./models/svc/vevosing/config/contentstyle_fvq16384_12.5hz_to_24kmels.json"
    )
    fmt_ckpt_path = os.path.join(
        local_dir, "acoustic_modeling/contentstyle_fvq16384_12.5hz_to_24kmels"
    )

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/VevoSing",
        repo_type="model",
        cache_dir="./ckpts/VevoSing",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )

    vocoder_cfg_path = "./models/svc/vevosing/config/vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # ===== Inference =====
    inference_pipeline = VevosingInferencePipeline(
        content_style_tokenizer_ckpt_path=tokenizer_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    content_wav_path = "./models/svc/vevosing/wav/jaychou.wav"
    reference_wav_path = "./models/svc/vevosing/wav/adele.wav"
    output_path = "./models/svc/vevosing/output/vevosing_svc.wav"

    vevosing_fm(content_wav_path, reference_wav_path, output_path)
