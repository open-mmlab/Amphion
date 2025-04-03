# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from huggingface_hub import snapshot_download

from models.svc.vevosing.vevosing_utils import *


def vevo_tts(
    tgt_text,
    ref_wav_path,
    timbre_ref_wav_path=None,
    output_path=None,
    ref_text=None,
    src_language="en",
    ref_language="en",
):
    if timbre_ref_wav_path is None:
        timbre_ref_wav_path = ref_wav_path

    gen_audio = inference_pipeline.inference_ar_and_fm(
        task="synthesis",
        src_wav_path=None,
        src_text=tgt_text,
        style_ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=timbre_ref_wav_path,
        style_ref_wav_text=ref_text,
        src_text_language=src_language,
        style_ref_wav_text_language=ref_language,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


if __name__ == "__main__":
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===== Prosody Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/VevoSing",
        repo_type="model",
        cache_dir="./ckpts/VevoSing",
        allow_patterns=["tokenizer/style_fvq512_6.25hz/*"],
    )
    prosody_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/style_fvq512_6.25hz"
    )

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/VevoSing",
        repo_type="model",
        cache_dir="./ckpts/VevoSing",
        allow_patterns=["tokenizer/contentstyle_fvq16384_12.5hz/*"],
    )
    contentstyle_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
    )

    # ===== Autoregressive Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/VevoSing",
        repo_type="model",
        cache_dir="./ckpts/VevoSing",
        allow_patterns=[
            "contentstyle_modeling/text_prosody_fvq512_6.25hz_to_contentstyle_fvq16384_12.5hz/*"
        ],
    )

    ar_cfg_path = "./models/svc/vevosing/config/text_prosody_fvq512_6.25hz_to_contentstyle_fvq16384_12.5hz.json"
    ar_ckpt_path = os.path.join(
        local_dir,
        "contentstyle_modeling/text_prosody_fvq512_6.25hz_to_contentstyle_fvq16384_12.5hz",
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
        prosody_tokenizer_ckpt_path=prosody_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=contentstyle_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    ### Zero-shot Text-to-Speech and Text-to-Singing  ###
    tgt_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."
    ref_wav_path = "./models/vc/vevo/wav/arabic_male.wav"
    ref_text = "Flip stood undecided, his ears strained to catch the slightest sound."

    # the style reference and timbre reference are same
    vevo_tts(
        tgt_text=tgt_text,
        ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=ref_wav_path,
        output_path="./models/svc/vevosing/output/zstts.wav",
        ref_text=ref_text,
        src_language="en",
        ref_language="en",
    )

    # the style reference and timbre reference are different
    vevo_tts(
        tgt_text=tgt_text,
        ref_wav_path=ref_wav_path,
        timbre_ref_wav_path="./models/vc/vevo/wav/mandarin_female.wav",
        output_path="./models/svc/vevosing/output/zstts_disentangled.wav",
        ref_text=ref_text,
        src_language="en",
        ref_language="en",
    )

    # TODO:
    # # the style reference is a singing voice
    # vevo_tts(
    #     tgt_text=tgt_text,
    #     ref_wav_path=ref_wav_path,
    #     timbre_ref_wav_path=ref_wav_path,
    #     output_path="./models/svc/vevosing/output/zstts_singing.wav",
    #     ref_text=ref_text,
    # )
