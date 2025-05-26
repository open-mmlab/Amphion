# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
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
args = parser.parse_args()

config = {}
args.ref_audio = (
    args.ref_audio or f"{package_dir}/infer/examples/basic/example_wav_en.wav"
)
args.gen_file = args.gen_file or "prediction.wav"

args.output_dir = args.output_dir or config.get("output_dir", "tests")
os.makedirs(args.output_dir, exist_ok=True)

args.output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)


def main():
    from dualcodec.model_tts.voicebox.voicebox_models import (
        voicebox_300M,
        extract_normalized_mel_spec_50hz,
    )
    from dualcodec.infer.voicebox.utils_voicebox_infer import (
        load_voicebox_300M_model,
        get_vocoder_decode_func_and_mel_spec,
        load_dualcodec_12hzv1_model,
        voicebox_inference,
    )

    voicebox_model_obj = load_voicebox_300M_model(device=device)

    vocoder_decode_func, mel_model = get_vocoder_decode_func_and_mel_spec(device=device)

    # extract GT dualcodec tokens
    dualcodec_inference_obj = load_dualcodec_12hzv1_model(device=device)
    import torchaudio

    # audio, sr = torchaudio.load(f"{package_dir}/infer/examples/basic/example_wav_en.wav")
    audio, sr = torchaudio.load(args.ref_audio)
    logger.info(f"loaded audio from {args.ref_audio}")

    # resample to 24kHz
    audio = torchaudio.functional.resample(audio, sr, 24000)
    audio = audio.reshape(1, 1, -1)
    audio = audio.to(device)
    # extract codes, for example, using 8 quantizers here:
    semantic_codes, acoustic_codes = dualcodec_inference_obj.encode(
        audio, n_quantizers=8
    )
    # semantic_codes shape: torch.Size([1, 1, T])
    # acoustic_codes shape: torch.Size([1, n_quantizers-1, T])

    # change semantic_codes to [b, t]
    semantic_codes = semantic_codes.squeeze(1)

    # use first 3s of acoustic codes as prompt
    audio = audio[:, :, : int(24000 * 2)]

    predicted = voicebox_inference(
        voicebox_model_obj=voicebox_model_obj,
        vocoder_decode_func=vocoder_decode_func,
        mel_spec_extractor_func=extract_normalized_mel_spec_50hz,
        combine_semantic_code=semantic_codes,
        prompt_speech=audio.squeeze(1),  # [b t]
    )

    out_path = os.path.join(args.output_dir, args.output_file)
    torchaudio.save(out_path, predicted.cpu(), 24000)
    logger.info(f"saved voicebox reconstruction audio to {out_path}")


if __name__ == "__main__":
    main()
