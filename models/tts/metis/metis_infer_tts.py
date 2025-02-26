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

    # TTS
    metis_cfg = load_config("./models/tts/metis/config/tts.json")

    ckpt_dir = snapshot_download(
        "amphion/maskgct",
        repo_type="model",
        local_dir="./models/tts/maskgct/ckpt",
        allow_patterns=["t2s/model.safetensors"],
    )

    ckpt_path = os.path.join(ckpt_dir, "t2s/model.safetensors")

    metis = Metis(
        ckpt_path=ckpt_path,
        cfg=metis_cfg,
        device=device,
        model_type="tts",
    )

    prompt_speech_path = "./models/tts/metis/wav/tts/prompt.wav"
    prompt_text = "Well, a growing number of people from ethnic backgrounds are getting bored of all these white male superheroes they can't relate to. And they're hungry for characters a little closer to home, or relevant to their own lives."
    text = "That’s true. But times have changed, and comic books these days often blur the line between right and wrong, making things unclear. Superheroes don't always do the thing and struggle with everyday problems like you and me."

    n_timesteps = 25
    cfg = 2.5
    model_type = "tts"

    gen_speech = metis(
        prompt_speech_path=prompt_speech_path,
        text=text,
        prompt_text=prompt_text,
        model_type=model_type,
        n_timesteps=n_timesteps,
        cfg=cfg,
    )

    sf.write("./models/tts/metis/wav/tts/gen.wav", gen_speech, 24000)
