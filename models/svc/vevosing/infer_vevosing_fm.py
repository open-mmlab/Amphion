# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from huggingface_hub import snapshot_download

from models.svc.vevosing.vevosing_utils import *


def vevosing_fm(content_wav_path, reference_wav_path, output_path, shifted_src=True):
    gen_audio = inference_pipeline.inference_fm(
        src_wav_path=content_wav_path,
        timbre_ref_wav_path=reference_wav_path,
        use_shifted_src_to_extract_prosody=shifted_src,
        flow_matching_steps=32,
    )
    save_audio(gen_audio, output_path=output_path)


def load_inference_pipeline():
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["tokenizer/contentstyle_fvq16384_12.5hz/*"],
    )
    contentstyle_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
    )

    # ===== Flow Matching Transformer =====
    model_name = "fm_emilia101k_singnet7k"

    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=[f"acoustic_modeling/{model_name}/*"],
    )

    fmt_cfg_path = f"./models/svc/vevosing/config/{model_name}.json"
    fmt_ckpt_path = os.path.join(local_dir, f"acoustic_modeling/{model_name}")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )

    vocoder_cfg_path = "./models/svc/vevosing/config/vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # ===== Inference =====
    inference_pipeline = VevosingInferencePipeline(
        content_style_tokenizer_ckpt_path=contentstyle_tokenizer_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )
    return inference_pipeline


if __name__ == "__main__":
    inference_pipeline = load_inference_pipeline()

    output_dir = "./models/svc/vevosing/output"
    os.makedirs(output_dir, exist_ok=True)

    content_wav_path = "./models/svc/vevosing/wav/jaychou.wav"
    reference_wav_path = "./models/svc/vevosing/wav/adele.wav"
    output_path = "./models/svc/vevosing/output/vevosing_svc.wav"

    vevosing_fm(content_wav_path, reference_wav_path, output_path)
