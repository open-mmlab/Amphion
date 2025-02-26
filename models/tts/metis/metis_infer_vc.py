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

    ########################## full-scale fine-tuned model ##########################

    metis_cfg = load_config("./models/tts/metis/config/ft.json")

    ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir="./models/tts/metis/ckpt",
        allow_patterns=["metis_vc/metis_vc.safetensors"],
    )

    ckpt_path = os.path.join(ckpt_dir, "metis_vc/metis_vc.safetensors")

    metis = Metis(
        ckpt_path=ckpt_path,
        cfg=metis_cfg,
        device=device,
        model_type="vc",
    )

    ##########################################################################################

    prompt_speech_path = "./models/tts/metis/wav/vc/prompt.wav"
    source_speech_path = "./models/tts/metis/wav/vc/source.wav"

    n_timesteps = 20
    cfg = 1.0

    gen_speech = metis(
        prompt_speech_path=prompt_speech_path,
        source_speech_path=source_speech_path,
        cfg=cfg,
        n_timesteps=n_timesteps,
        model_type="vc",
    )

    sf.write("./models/tts/metis/wav/vc/gen.wav", gen_speech, 24000)
