import os
import sys
import copy
import json
import time
import random
import argparse
import soundfile as sf
import numpy as np
import librosa
import torchaudio
from tqdm import tqdm
import laion_clap
from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
from sklearn.metrics.pairwise import cosine_similarity

import torch
from datetime import datetime
from diffusers import DDPMScheduler
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

import models.controllable_diffusion as ConDiffusion
import models.controllable_dataset as ConDataset
from data.filter_data import get_event_list


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for text to audio generation task."
    )
    parser.add_argument(
        "--exp_path", "-exp", type=str, default=None, help="Path for experiment."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/meta_data/test-onoff-control_multi-event.json",
        help="Path for test_file.",
    )
    parser.add_argument(
        "--original_args",
        type=str,
        default="summary.jsonl",
        help="Path for summary jsonl file saved during training.",
    )
    parser.add_argument(
        "--model_pt", type=str, default="best.pt", help="Path for saved model bin file."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance",
        "-g",
        type=float,
        # default=3,
        default=1,
        help="Guidance scale for classifier free guidance.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed.",
    )

    args = parser.parse_args()

    args.original_args = os.path.join(args.exp_path, args.original_args)
    args.model_pt = os.path.join(args.exp_path, args.model_pt)
    return args


def main():
    args = parse_args()
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Prepare Data #
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files={"test": args.test_file})
    test_dataset = getattr(ConDataset, train_args.dataset_class)(
        raw_datasets["test"], train_args
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_dataset.collate_fn,
    )

    # Load Models #
    print("\n------Load model")
    name = "audioldm-s-full"
    vae, stft = ConDiffusion.build_pretrained_models(name)
    vae, stft = vae.cuda(), stft.cuda()
    print(train_args.model_class)
    model = (
        getattr(ConDiffusion, train_args.model_class)(
            scheduler_name=train_args.scheduler_name,
            unet_model_config_path=train_args.unet_model_config,
            snr_gamma=train_args.snr_gamma,
        )
        .cuda()
        .eval()
    )

    # Load Trained Weight #
    device = vae.device()
    model.load_state_dict(torch.load(args.model_pt))
    scheduler = DDPMScheduler.from_pretrained(
        train_args.scheduler_name, subfolder="scheduler"
    )

    # Generate #
    num_steps, guidance, batch_size, num_samples = (
        args.num_steps,
        args.guidance,
        args.batch_size,
        args.num_samples,
    )
    audio_len = 16000 * 10
    output_dir = os.path.join(
        CONTROLLABLE_PATH,
        f"synthesized/{'-'.join(args.model_pt.split('/')[-3:-1])}_steps-{num_steps}_guidance-{guidance}_samples-{num_samples}_{args.test_file.split('/')[-1].split('.')[0]}/",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"------Write to files to {output_dir}")

    print("------Diffusion begin!")
    if args.num_samples == 1:
        for batch in tqdm(test_dataloader):
            idx, onset, event_info, _, _, onset_str = (
                batch  # idx, onset, event_info, audios, caption, onset_str
            )
            with torch.no_grad():
                latents = model.inference(
                    {"onset": onset.to(device), "event_info": event_info.to(device)},
                    scheduler,
                    num_steps,
                    guidance,
                    num_samples,
                    disable_progress=True,
                )
                mel = vae.decode_first_stage(latents)
                wave = vae.decode_to_waveform(mel)
            for j, wav in enumerate(wave):
                sf.write(
                    f"{output_dir}/{idx[j]}--{onset_str[j]}.wav",
                    wav[:audio_len],
                    samplerate=16000,
                    subtype="PCM_16",
                )
    else:
        print("Clap scorer filter")
        clap_scorer = laion_clap.CLAP_Module(enable_fusion=False)
        ckpt_path = "miniconda3/envs/py3.10.11/lib/python3.10/site-packages/laion_clap/630k-audioset-best.pt"
        ckpt = clap_load_state_dict(ckpt_path, skip_params=True)
        del_parameter_key = ["text_branch.embeddings.position_ids"]
        ckpt = {"model." + k: v for k, v in ckpt.items() if k not in del_parameter_key}
        clap_scorer.load_state_dict(ckpt)
        for batch in tqdm(test_dataloader):
            _, onset, event_info, _, caption, onset_str = (
                batch  # idx, onset, event_info, audios, caption, onset_str
            )
            with torch.no_grad():
                latents = model.inference(
                    {"onset": onset.to(device), "event_info": event_info.to(device)},
                    scheduler,
                    num_steps,
                    guidance,
                    num_samples,
                    disable_progress=True,
                )
                mel = vae.decode_first_stage(latents)
                wave = vae.decode_to_waveform(mel)
            for j in range(batch_size):
                text_embed = clap_scorer.get_text_embedding(
                    [caption[j], ""], use_tensor=False
                )[:1]
                best_idx, best_score = 0, float("-inf")
                for candidate_idx in range(num_samples):
                    wav_48k = librosa.core.resample(
                        wave[j * num_samples + candidate_idx].astype(np.float64)[
                            :audio_len
                        ]
                        / 32768,
                        orig_sr=16000,
                        target_sr=48000,
                    ).reshape(1, -1)
                    audio_embed = clap_scorer.get_audio_embedding_from_data(x=wav_48k)
                    pair_similarity = cosine_similarity(audio_embed, text_embed)[0][0]
                    if pair_similarity > best_score:
                        best_score = pair_similarity
                        best_idx = candidate_idx
                sf.write(
                    f"{output_dir}/{idx[j]}--{onset_str[j]}.wav",
                    wave[j * num_samples + best_idx][:audio_len],
                    samplerate=16000,
                    subtype="PCM_16",
                )


if __name__ == "__main__":
    main()
