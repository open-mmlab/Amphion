# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import tempfile
from collections import OrderedDict

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
import hydra

from loguru import logger

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


# from dualcodec.model_tts.valle_ar import ValleARInference
from dualcodec.utils.utils_infer import (
    load_checkpoint,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


# load models
def load_dualcodec_valle_12hzv1():
    TTS_MODEL_CFG = {
        "model": "valle_ar",
        "ckpt_path": "hf://amphion/dualcodec-tts/dualcodec_valle_ar_12hzv1.safetensors",
        # "ckpt_path": "dualcodec_tts_ckpts/dualcodec_valle_ar_12hzv1.safetensors",
        "cfg_path": "conf_tts/model/valle_ar/llama_250M.yaml",
    }
    model_cfg_path = TTS_MODEL_CFG["cfg_path"]
    # instantiate model
    with hydra.initialize(config_path=model_cfg_path):
        cfg = hydra.compose(config_name=model_cfg_path)
    model = hydra.utils.instantiate(cfg.model)
    ckpt_path = TTS_MODEL_CFG["ckpt_path"]
    load_checkpoint(model, ckpt_path, use_ema=False)
    return model


VALLE_model = load_dualcodec_valle_12hzv1()


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence=False,
    cross_fade_duration=0.15,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(
        ref_audio_orig, ref_text, show_info=show_info
    )

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(
                model[1], vocab_path=model[2], model_cfg=model[3]
            )
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


if __name__ == "__main__":
    load_dualcodec_valle_12hzv1()
