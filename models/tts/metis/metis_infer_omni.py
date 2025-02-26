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

    # Omni
    metis_cfg = load_config("./models/tts/metis/config/omni.json")

    ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir="./models/tts/metis/ckpt",
        allow_patterns=["metis_omni/metis_omni.safetensors"],
    )

    ckpt_path = os.path.join(ckpt_dir, "metis_omni/metis_omni.safetensors")

    metis = Metis(
        ckpt_path=ckpt_path,
        cfg=metis_cfg,
        device=device,
        model_type="omni",
    )

    # infer for TTS
    prompt_speech_path = "./models/tts/metis/wav/tts/prompt.wav"
    prompt_text = "Well, a growing number of people from ethnic backgrounds are getting bored of all these white male superheroes they can't relate to. And they're hungry for characters a little closer to home, or relevant to their own lives."
    text = "That’s true. But times have changed, and comic books these days often blur the line between right and wrong, making things unclear. Superheroes don't always do the thing and struggle with everyday problems like you and me."

    gen_speech = metis(
        prompt_speech_path=prompt_speech_path,
        text=text,
        prompt_text=prompt_text,
        model_type="tts",
        n_timesteps=25,
        cfg=2.5,
    )

    sf.write("./models/tts/metis/wav/tts/omni_gen.wav", gen_speech, 24000)

    # infer for TSE
    prompt_speech_path = "./models/tts/metis/wav/tse/prompt.wav"
    source_speech_path = "./models/tts/metis/wav/tse/mix.wav"

    gen_speech = metis(
        prompt_speech_path=prompt_speech_path,
        source_speech_path=source_speech_path,
        cfg=0.0,
        n_timesteps=10,
        model_type="tse",
    )

    sf.write("./models/tts/metis/wav/tse/omni_gen.wav", gen_speech, 24000)

    # infer for VC
    prompt_speech_path = "./models/tts/metis/wav/vc/prompt.wav"
    source_speech_path = "./models/tts/metis/wav/vc/source.wav"

    gen_speech = metis(
        prompt_speech_path=prompt_speech_path,
        source_speech_path=source_speech_path,
        cfg=1.0,
        n_timesteps=20,
        model_type="vc",
    )

    sf.write("./models/tts/metis/wav/vc/omni_gen.wav", gen_speech, 24000)

    # infer for SE
    source_speech_path = "./models/tts/metis/wav/se/noise.wav"

    gen_speech = metis(
        source_speech_path=source_speech_path,
        cfg=0.0,
        n_timesteps=10,
        model_type="se",
    )

    sf.write("./models/tts/metis/wav/se/omni_gen.wav", gen_speech, 24000)
