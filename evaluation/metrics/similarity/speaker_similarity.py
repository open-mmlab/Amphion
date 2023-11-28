# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
import librosa

from .models.RawNetModel import RawNet3
from .models.RawNetBasicBlock import Bottle2neck


def extract_speaker_embd(
    model, fn: str, n_samples: int, n_segments: int = 10, gpu: bool = False
) -> np.ndarray:
    audio, sample_rate = sf.read(fn)
    if len(audio.shape) > 1:
        raise ValueError(
            f"RawNet3 supports mono input only. Input data has a shape of {audio.shape}."
        )

    if sample_rate != 16000:
        # resample to 16000kHz
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        # print("resample to 16000kHz!")
    if len(audio) < n_samples:  # RawNet3 was trained using utterances of 3 seconds
        shortage = n_samples - len(audio) + 1
        audio = np.pad(audio, (0, shortage), "wrap")

    audios = []
    startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
    for asf in startframe:
        audios.append(audio[int(asf) : int(asf) + n_samples])

    audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32))
    if gpu:
        audios = audios.to("cuda")
    with torch.no_grad():
        output = model(audios)

    return output


def extract_speaker_similarity(target_path, reference_path):
    model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )

    gpu = False
    model.load_state_dict(
        torch.load(
            "pretrained/rawnet3/model.pt",
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    model.eval()
    print("RawNet3 initialised & weights loaded!")

    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True
    # for target_path, reference_path in zip(target_paths, ref_paths):
    # print(f"Extracting embeddings for target singers...")

    target_embeddings = []
    for file in tqdm(os.listdir(target_path)):
        output = extract_speaker_embd(
            model,
            fn=os.path.join(target_path, file),
            n_samples=48000,
            n_segments=10,
            gpu=gpu,
        ).mean(0)
        target_embeddings.append(output.detach().cpu().numpy())
    target_embeddings = np.array(target_embeddings)
    target_embedding = np.mean(target_embeddings, axis=0)

    # print(f"Extracting embeddings for reference singer...")

    reference_embeddings = []
    for file in tqdm(os.listdir(reference_path)):
        output = extract_speaker_embd(
            model,
            fn=os.path.join(reference_path, file),
            n_samples=48000,
            n_segments=10,
            gpu=gpu,
        ).mean(0)
        reference_embeddings.append(output.detach().cpu().numpy())
    reference_embeddings = np.array(reference_embeddings)

    # print("Calculating cosine similarity...")

    cos_sim = F.cosine_similarity(
        torch.from_numpy(np.mean(target_embeddings, axis=0)).unsqueeze(0),
        torch.from_numpy(np.mean(reference_embeddings, axis=0)).unsqueeze(0),
        dim=1,
    )

    # print(f"Mean cosine similarity: {cos_sim.item()}")

    return cos_sim.item()
