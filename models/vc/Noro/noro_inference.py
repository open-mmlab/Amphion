# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import numpy as np
import librosa
from safetensors.torch import load_model
import os
from utils.util import load_config
from models.vc.Noro.noro_trainer import NoroTrainer
from models.vc.Noro.noro_model import Noro_VCmodel
from processors.content_extractor import HubertExtractor
from utils.mel import mel_spectrogram_torch
from utils.f0 import get_f0_features_using_dio, interpolate
from torch.nn.utils.rnn import pad_sequence


def build_trainer(args, cfg):
    supported_trainer = {
        "VC": NoroTrainer,
    }
    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="JSON file for configurations.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Checkpoint for resume training or fine-tuning.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Output path",
        required=True,
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        help="Reference voice path",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        help="Source voice path",
    )
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA id for training.")

    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cuda_id = args.cuda_id
    args.local_rank = torch.device(f"cuda:{cuda_id}")
    print("Local rank:", args.local_rank)

    args.content_extractor = "mhubert"

    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()
    ckpt_path = args.checkpoint_path

    w2v = HubertExtractor(cfg)
    w2v = w2v.to(device=args.local_rank)
    w2v.eval()

    model = Noro_VCmodel(cfg=cfg.model)
    print("Loading model")

    load_model(model, ckpt_path)
    print("Model loaded")
    model.cuda(args.local_rank)
    model.eval()

    wav_path = args.source_path
    ref_wav_path = args.ref_path

    wav, _ = librosa.load(wav_path, sr=16000)
    wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
    audio = torch.from_numpy(wav).to(args.local_rank)
    audio = audio[None, :]

    ref_wav, _ = librosa.load(ref_wav_path, sr=16000)
    ref_wav = np.pad(ref_wav, (0, 200 - len(ref_wav) % 200))
    ref_audio = torch.from_numpy(ref_wav).to(args.local_rank)
    ref_audio = ref_audio[None, :]

    with torch.no_grad():
        ref_mel = mel_spectrogram_torch(ref_audio, cfg)
        ref_mel = ref_mel.transpose(1, 2).to(device=args.local_rank)
        ref_mask = (
            torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(args.local_rank).bool()
        )

        _, content_feature = w2v.extract_content_features(audio)
        content_feature = content_feature.to(device=args.local_rank)

        wav = audio.cpu().numpy()
        wav = wav[0, :]
        f0s = []
        pitch_raw = get_f0_features_using_dio(wav, cfg.preprocess)
        pitch_raw, _ = interpolate(pitch_raw)
        frame_num = len(wav) // cfg.preprocess.hop_size
        pitch_raw = torch.from_numpy(pitch_raw[:frame_num]).float()
        f0s.append(pitch_raw)
        pitch = pad_sequence(f0s, batch_first=True, padding_value=0).float()
        pitch = (pitch - pitch.mean(dim=1, keepdim=True)) / (
            pitch.std(dim=1, keepdim=True) + 1e-6
        )
        pitch = pitch.to(device=args.local_rank)

        x0 = model.inference(
            content_feature=content_feature,
            pitch=pitch,
            x_ref=ref_mel,
            x_ref_mask=ref_mask,
            inference_steps=200,
            sigma=1.2,
        )  # 150-300 0.95-1.5

        recon_path = f"{args.output_dir}/recon_mel.npy"
        np.save(recon_path, x0.transpose(1, 2).detach().cpu().numpy())
        print(f"Mel spectrogram saved to: {recon_path}")


if __name__ == "__main__":
    main()
