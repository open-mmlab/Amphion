# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchaudio
import torch
import os
import json5
import dac
from models.se.anyenhance.modules.anyenhance_modules import MaskGitTransformer
from models.se.anyenhance.anyenhance_model import AudioEncoder_v2, AnyEnhance

os.environ["PYTHONIOENCODING"] = "utf-8"

task_type_map = {"enhancement": 0, "extraction": 1}


def pad_or_truncate(x, length=512 * 256):
    if x.size(-1) < length:
        repeat_times = length // x.size(-1) + 1
        x = x.repeat(1, repeat_times)
        x = x[..., :length]
    elif x.size(-1) > length:
        x = x[..., :length]
    return x


def get_model(dac_path, config, device):
    # Load DAC model
    dac_model = dac.DAC.load(dac_path).to(device)
    dac_model.to(device)
    dac_model.eval()
    dac_model.requires_grad_(False)

    # Initialize transformer and audio encoder
    transformer_config = config["MaskGitTransformer"]
    audio_encoder_config = config["AudioEncoder"]
    transformer = MaskGitTransformer(**transformer_config)
    audio_encoder = AudioEncoder_v2(**audio_encoder_config)

    anyenhance_config = config["AnyEnhance"]
    model = AnyEnhance(
        vq_model=dac_model,
        transformer=transformer,
        audio_encoder=audio_encoder,
        **anyenhance_config,
    ).to(device)

    print(
        f"model Params: {round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2)}M"
    )

    return model


def load_model(model_path, dac_path, config, device):
    model_state_dict = torch.load(model_path)
    if "module" in list(model_state_dict.keys())[0]:
        print("New model detected. Loading new model.")
        model = get_model(
            dac_path, config["model"], device
        )  # get_model needs to be defined or imported appropriately
        model = torch.nn.DataParallel(model)
        model.load_state_dict(model_state_dict, strict=False)
        model = model.module
    else:
        print("No MODULE.")
        model = get_model(dac_path, config["model"], device)
        model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model


def smooth_audio_transition(audio_chunks, overlap=1024):
    """
    Smoothly transition between audio chunks by hann windowing the overlap region.
    """
    if len(audio_chunks) == 0:
        return torch.Tensor([])

    window = torch.hann_window(overlap * 2, periodic=True).to(audio_chunks[0].device)

    result = audio_chunks[0]

    for i in range(1, len(audio_chunks)):
        prev_chunk_length = result.shape[-1]
        prev_overlap_len = min(overlap, prev_chunk_length)

        previous_end = result[:, -prev_overlap_len:].clone()

        curr_chunk_length = audio_chunks[i].shape[-1]
        curr_overlap_len = min(overlap, curr_chunk_length)

        previous_end *= (
            window[overlap : overlap + prev_overlap_len]
            if prev_overlap_len == overlap
            else window[-prev_overlap_len:]
        )
        current_start = audio_chunks[i][:, :curr_overlap_len].clone()
        current_start *= window[:curr_overlap_len]

        min_len = min(prev_overlap_len, curr_overlap_len)
        transition = previous_end[:, -min_len:] + current_start[:, :min_len]

        result[:, -min_len:] = transition

        result = torch.cat((result, audio_chunks[i][:, curr_overlap_len:]), dim=1)

    return result


def process_single_audio(
    model,
    signal,
    device,
    task_type,
    window_size,
    overlap=1024,
    prompt_signal=None,
    timesteps=20,
    cond_scale=1,
    force_not_use_token_critic=False,
):
    """
    Enhance a single audio signal based on the task type and optional prompt.
    """
    infer_task_type = torch.tensor([task_type_map[task_type]], dtype=torch.long).to(
        device
    )

    original_length = signal.shape[-1]
    enhanced_audio = []
    start = 0
    hop_size = window_size - overlap

    while start < signal.shape[-1]:
        if start + window_size >= signal.shape[-1]:
            segment = signal[:, -window_size:]
            is_last_segment = True
        else:
            end = start + window_size
            segment = signal[:, start:end]
            is_last_segment = False

        if segment.shape[-1] < window_size:
            is_padding = True
            valid_length = segment.shape[-1]
            segment = torch.nn.functional.pad(segment, (0, window_size - valid_length))
        else:
            is_padding = False

        with torch.no_grad():
            if prompt_signal is not None:
                ids, output_segment = model.generate_with_prompt(
                    segment.unsqueeze(0),
                    prompt_signal.unsqueeze(0),
                    timesteps=timesteps,
                    cond_scale=cond_scale,
                    task_type=infer_task_type,
                    force_not_use_token_critic=force_not_use_token_critic,
                )
            else:
                ids, output_segment = model.generate(
                    segment.unsqueeze(0),
                    timesteps=timesteps,
                    cond_scale=cond_scale,
                    task_type=infer_task_type,
                    force_not_use_token_critic=force_not_use_token_critic,
                )
            output_segment = output_segment.squeeze(0)

            if is_last_segment:
                if is_padding:
                    output_segment = output_segment[:, :valid_length]
                last_valid_length = signal.shape[-1] - start
                if last_valid_length == 0:
                    last_valid_length = window_size
                output_segment = output_segment[:, -last_valid_length:]

            enhanced_audio.append(output_segment)

        start += hop_size

    enhanced_signal = smooth_audio_transition(enhanced_audio, overlap=overlap)
    enhanced_signal = enhanced_signal[:, :original_length]

    return enhanced_signal


def infer_single_audio(
    model,
    audio_path,
    output_path,
    device,
    task_type="enhancement",
    prompt_audio_path=None,
    timesteps=20,
    cond_scale=1,
    force_not_use_token_critic=False,
):
    assert (
        task_type == "enhancement" or prompt_audio_path is not None
    ), "Prompt audio path must be provided for extraction task."

    os.makedirs(output_path, exist_ok=True)

    signal, sr = torchaudio.load(audio_path)
    signal = torch.mean(signal, dim=0, keepdim=True)
    signal = signal.to(device)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100).to(device)
    signal = resampler(signal)

    if prompt_audio_path is not None:
        prompt_signal, prompt_sr = torchaudio.load(prompt_audio_path)
        prompt_signal = torch.mean(prompt_signal, dim=0, keepdim=True)
        prompt_signal = prompt_signal.to(device)
        prompt_resampler = torchaudio.transforms.Resample(
            orig_freq=prompt_sr, new_freq=44100
        ).to(device)
        prompt_signal = prompt_resampler(prompt_signal)
        prompt_signal = pad_or_truncate(prompt_signal, model.prompt_len * 512)
    else:
        prompt_signal = None

    window_size = (
        512 * model.seq_len
        if prompt_audio_path is not None
        else 512 * (model.seq_len + model.prompt_len)
    )
    overlap = 1024

    enhanced_signal = process_single_audio(
        model,
        signal,
        device,
        task_type,
        window_size=window_size,
        overlap=overlap,
        prompt_signal=prompt_signal,
        timesteps=timesteps,
        cond_scale=cond_scale,
        force_not_use_token_critic=force_not_use_token_critic,
    )

    output_file_path = os.path.join(output_path, os.path.basename(audio_path))
    torchaudio.save(output_file_path, enhanced_signal.detach().cpu(), 44100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate enhanced audio outputs from given audio files."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to audio file to process."
    )
    parser.add_argument(
        "--task_type", type=str, required=True, help="Path to audio file to process."
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to the output folder."
    )
    parser.add_argument(
        "--prompt_file", type=str, default=None, help="Path to prompt audio file."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on."
    )
    parser.add_argument(
        "--timesteps", type=int, default=20, help="Number of timesteps to generate"
    )
    parser.add_argument("--cond_scale", type=float, default=1, help="CFG factor.")
    parser.add_argument(
        "--force_not_use_token_critic",
        action="store_true",
        help="Force not use token critic for inference.",
    )

    args = parser.parse_args()

    file_path = os.path.dirname(os.path.abspath(__file__))
    dac_path = os.path.join(file_path, "pretrained", "anyenhance", "dac", "weights.pth")
    model_path = os.path.join(
        file_path,
        "pretrained",
        "anyenhance",
        "epoch-1-step-300000-loss-4.3083",
        "model.pt",
    )
    config_path = os.path.join(
        file_path, "pretrained", "anyenhance", "anyenhance-360M-selfcritic-v2.json"
    )

    print(f"Using model: {model_path}")

    model = load_model(model_path, dac_path, json5.load(open(config_path)), args.device)
    infer_single_audio(
        model,
        args.input_file,
        args.output_folder,
        args.device,
        args.task_type,
        args.prompt_file,
        args.timesteps,
        args.cond_scale,
        args.force_not_use_token_critic,
    )
