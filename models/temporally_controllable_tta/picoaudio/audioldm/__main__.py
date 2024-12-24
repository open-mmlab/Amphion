#!/usr/bin/python3
import os
from audioldm import text_to_audio, style_transfer, build_model, save_wave, get_time, round_up_duration, get_duration
import argparse

CACHE_DIR = os.getenv(
    "AUDIOLDM_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/audioldm"))

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    type=str,
    required=False,
    default="generation",
    help="generation: text-to-audio generation; transfer: style transfer",
    choices=["generation", "transfer"]
)

parser.add_argument(
    "-t",
    "--text",
    type=str,
    required=False,
    default="",
    help="Text prompt to the model for audio generation",
)

parser.add_argument(
    "-f",
    "--file_path",
    type=str,
    required=False,
    default=None,
    help="(--mode transfer): Original audio file for style transfer; Or (--mode generation): the guidance audio file for generating simialr audio",
)

parser.add_argument(
    "--transfer_strength",
    type=float,
    required=False,
    default=0.5,
    help="A value between 0 and 1. 0 means original audio without transfer, 1 means completely transfer to the audio indicated by text",
)

parser.add_argument(
    "-s",
    "--save_path",
    type=str,
    required=False,
    help="The path to save model output",
    default="./output",
)

parser.add_argument(
    "--model_name",
    type=str,
    required=False,
    help="The checkpoint you gonna use",
    default="audioldm-s-full",
    choices=["audioldm-s-full", "audioldm-l-full", "audioldm-s-full-v2"]
)

parser.add_argument(
    "-ckpt",
    "--ckpt_path",
    type=str,
    required=False,
    help="The path to the pretrained .ckpt model",
    default=None,
)

parser.add_argument(
    "-b",
    "--batchsize",
    type=int,
    required=False,
    default=1,
    help="Generate how many samples at the same time",
)

parser.add_argument(
    "--ddim_steps",
    type=int,
    required=False,
    default=200,
    help="The sampling step for DDIM",
)

parser.add_argument(
    "-gs",
    "--guidance_scale",
    type=float,
    required=False,
    default=2.5,
    help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
)

parser.add_argument(
    "-dur",
    "--duration",
    type=float,
    required=False,
    default=10.0,
    help="The duration of the samples",
)

parser.add_argument(
    "-n",
    "--n_candidate_gen_per_text",
    type=int,
    required=False,
    default=3,
    help="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
)

parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default=42,
    help="Change this value (any integer number) will lead to a different generation result.",
)

args = parser.parse_args()

if(args.ckpt_path is not None):
    print("Warning: ckpt_path has no effect after version 0.0.20.")
    
assert args.duration % 2.5 == 0, "Duration must be a multiple of 2.5"

mode = args.mode
if(mode == "generation" and args.file_path is not None):
    mode = "generation_audio_to_audio"
    if(len(args.text) > 0):
        print("Warning: You have specified the --file_path. --text will be ignored")
        args.text = ""
        
save_path = os.path.join(args.save_path, mode)

if(args.file_path is not None):
    save_path = os.path.join(save_path, os.path.basename(args.file_path.split(".")[0]))

text = args.text
random_seed = args.seed
duration = args.duration
guidance_scale = args.guidance_scale
n_candidate_gen_per_text = args.n_candidate_gen_per_text

os.makedirs(save_path, exist_ok=True)
audioldm = build_model(model_name=args.model_name)

if(args.mode == "generation"):
    waveform = text_to_audio(
        audioldm,
        text,
        args.file_path,
        random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        ddim_steps=args.ddim_steps,
        n_candidate_gen_per_text=n_candidate_gen_per_text,
        batchsize=args.batchsize,
    )
    
elif(args.mode == "transfer"):
    assert args.file_path is not None
    assert os.path.exists(args.file_path), "The original audio file \'%s\' for style transfer does not exist." % args.file_path
    waveform = style_transfer(
        audioldm,
        text,
        args.file_path,
        args.transfer_strength,
        random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        ddim_steps=args.ddim_steps,
        batchsize=args.batchsize,
    )
    waveform = waveform[:,None,:]

save_wave(waveform, save_path, name="%s_%s" % (get_time(), text))
