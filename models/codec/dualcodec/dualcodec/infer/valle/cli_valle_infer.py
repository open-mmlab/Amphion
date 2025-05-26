# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from omegaconf import OmegaConf
from loguru import logger
from cached_path import cached_path
import hydra
from pathlib import Path

from dualcodec.utils.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    load_model,
    load_vocoder,
    load_checkpoint,
    instantiate_model,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    device,
    package_dir,
)
from dualcodec.infer.valle.utils_valle_infer import (
    infer_process,
    load_dualcodec_valle_nar_12hzv1,
    load_dualcodec_valle_ar_12hzv1,
)

parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
args = parser.parse_args()


# config file

# config = tomli.load(open(args.config, "rb"))
config = {}


logger.info("Loading Valle models...")
ar_model = load_dualcodec_valle_ar_12hzv1()
nar_model = load_dualcodec_valle_nar_12hzv1()
from dualcodec.utils import get_whisper_tokenizer

tokenizer_model = get_whisper_tokenizer()
import dualcodec

dualcodec_model = dualcodec.get_model("12hz_v1")
dualcodec_inference_obj = dualcodec.Inference(
    dualcodec_model=dualcodec_model, device=device, autocast=True
)
logger.info("Valle models loaded.")


ref_audio = args.ref_audio or config.get(
    "ref_audio", f"{package_dir}/infer/examples/basic/example_wav_en.wav"
)
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature. Others call me mother nature.")
)
gen_text = args.gen_text or config.get(
    "gen_text", "Here we generate something just for test."
)
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
remove_silence = args.remove_silence or config.get("remove_silence", False)

cross_fade_duration = args.cross_fade_duration or config.get(
    "cross_fade_duration", cross_fade_duration
)

# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

# if vocoder_name == "vocos":
#     vocoder_local_path = "../checkpoints/vocos-mel-24khz"
# elif vocoder_name == "bigvgan":
#     vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

# vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path)
vocoder = None
# inference process


def main():
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = (
            preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, spectrogram = infer_process(
            ar_model_obj=ar_model,
            nar_model_obj=nar_model,
            dualcodec_inference_obj=dualcodec_inference_obj,
            tokenizer_obj=tokenizer_model,
            ref_audio=ref_audio_,
            ref_text=ref_text_,
            gen_text=gen_text_,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            streaming=False,
            top_k=15,
            top_p=0.85,
            temperature=1.0,
            repeat_penalty=1.1,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            sf.write(
                os.path.join(
                    output_chunk_dir,
                    f"{len(generated_audio_segments)-1}_{gen_text_}.wav",
                ),
                audio_segment,
                final_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


if __name__ == "__main__":
    main()
