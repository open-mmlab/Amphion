# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from models.tts.metis.metis import Metis

from huggingface_hub import snapshot_download
from utils.util import load_config
import soundfile as sf

if __name__ == "__main__":

    device = "cuda:0"

    # SE
    metis_cfg = load_config("./models/tts/metis/config/se.json")

    # download model checkpoints.

    ########################## lora-based fine-tuned model ##########################

    # download base model, lora weights, and adapter weights
    base_ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir="./models/tts/metis/ckpt",
        allow_patterns=["metis_base/model.safetensors"],
    )
    lora_ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir="./models/tts/metis/ckpt",
        allow_patterns=["metis_se/metis_se_lora_32.safetensors"],
    )
    adapter_ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir="./models/tts/metis/ckpt",
        allow_patterns=["metis_se/metis_se_lora_32_adapter.safetensors"],
    )

    base_ckpt_path = os.path.join(base_ckpt_dir, "metis_base/model.safetensors")
    lora_ckpt_path = os.path.join(
        lora_ckpt_dir, "metis_se/metis_se_lora_32.safetensors"
    )
    adapter_ckpt_path = os.path.join(
        adapter_ckpt_dir, "metis_se/metis_se_lora_32_adapter.safetensors"
    )

    metis = Metis(
        base_ckpt_path=base_ckpt_path,
        lora_ckpt_path=lora_ckpt_path,
        adapter_ckpt_path=adapter_ckpt_path,
        cfg=metis_cfg,
        device=device,
        model_type="se",
    )

    ##########################################################################################

    source_speech_path = "./models/tts/metis/wav/se/noise.wav"

    n_timesteps = 10
    cfg = 0.0

    gen_speech = metis(
        source_speech_path=source_speech_path,
        cfg=cfg,
        n_timesteps=n_timesteps,
        model_type="se",
    )

    sf.write("./models/tts/metis/wav/se/gen.wav", gen_speech, 24000)
