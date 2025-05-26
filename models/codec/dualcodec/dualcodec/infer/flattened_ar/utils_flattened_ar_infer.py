# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dualcodec.infer.valle.utils_valle_infer import *


def load_flattened_ar_model():
    TTS_MODEL_CFG = {
        "model": "flattened_ar",
        "ckpt_path": "hf://amphion/dualcodec-tts/flattened_ar_dualcodec5x4096.safetensors",
        # "ckpt_path": "dualcodec_tts_ckpts/dualcodec_flattened_ar_12hzv1.safetensors",
        "cfg_path": "../conf_tts/model/flattened_ar/llama_1x32768.yaml",
    }
    model = (
        instantiate_model(
            model_cfg_path=TTS_MODEL_CFG["cfg_path"],
        )
        .half()
        .eval()
    )
    ckpt_path = TTS_MODEL_CFG["ckpt_path"]
    load_checkpoint(
        model,
        ckpt_path,
        use_ema=False,
        device=device,
    )
    return model


def get_flattened_ar_inference_obj(flattened_ar_model, dualcodec_inference_obj, device):
    from dualcodec.infer.flattened_ar.inference_flattened import Inference
    from dualcodec.utils.utils import get_whisper_tokenizer

    return Inference(
        model=flattened_ar_model,
        tokenizer_obj=get_whisper_tokenizer(),
        dualcodec_inference_obj=dualcodec_inference_obj,
        normalize=True,
        device=device,
    )
