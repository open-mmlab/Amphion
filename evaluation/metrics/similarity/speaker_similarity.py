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

from evaluation.metrics.similarity.models.RawNetModel import RawNet3
from evaluation.metrics.similarity.models.RawNetBasicBlock import Bottle2neck

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from resemblyzer import VoiceEncoder, preprocess_wav


def extract_rawnet_speaker_embd(
    model, fn: str, n_samples: int, n_segments: int = 10, gpu: bool = False
) -> np.ndarray:
    audio, sample_rate = sf.read(fn)
    if len(audio.shape) > 1:
        raise ValueError(
            f"RawNet3 supports mono input only. Input data has a shape of {audio.shape}."
        )

    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    if len(audio) < n_samples:
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


def extract_similarity(path_ref, path_deg, **kwargs):
    kwargs = kwargs["kwargs"]
    model_name = kwargs["model_name"]

    ref_embds = []
    deg_embds = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_name == "rawnet":
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
        model.load_state_dict(
            torch.load(
                "pretrained/rawnet3/model.pt",
                map_location=lambda storage, loc: storage,
            )["model"]
        )
        model.eval()
        model = model.to(device)

        for file in tqdm(os.listdir(path_ref)):
            output = extract_rawnet_speaker_embd(
                model,
                fn=os.path.join(path_ref, file),
                n_samples=48000,
                n_segments=10,
                gpu=torch.cuda.is_available(),
            ).mean(0)
            ref_embds.append(output)

        for file in tqdm(os.listdir(path_deg)):
            output = extract_rawnet_speaker_embd(
                model,
                fn=os.path.join(path_deg, file),
                n_samples=48000,
                n_segments=10,
                gpu=torch.cuda.is_available(),
            ).mean(0)
            deg_embds.append(output)
    elif model_name == "wavlm":
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus-sv"
            )
            model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        except:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "pretrained/wavlm", sampling_rate=16000
            )
            model = WavLMForXVector.from_pretrained("pretrained/wavlm")
        model = model.to(device)

        for file in tqdm(os.listdir(path_ref)):
            wav_path = os.path.join(path_ref, file)
            wav, _ = librosa.load(wav_path, sr=16000)

            inputs = feature_extractor(
                [wav], padding=True, return_tensors="pt", sampling_rate=16000
            )
            if torch.cuda.is_available():
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(device)

            with torch.no_grad():
                embds = model(**inputs).embeddings
                embds = embds
                ref_embds.append(embds[0])

        for file in tqdm(os.listdir(path_deg)):
            wav_path = os.path.join(path_deg, file)
            wav, _ = librosa.load(wav_path, sr=16000)

            inputs = feature_extractor(
                [wav], padding=True, return_tensors="pt", sampling_rate=16000
            )
            if torch.cuda.is_available():
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(device)

            with torch.no_grad():
                embds = model(**inputs).embeddings
                embds = embds
                deg_embds.append(embds[0])
    elif model_name == "resemblyzer":
        encoder = VoiceEncoder().to(device)

        for file in tqdm(os.listdir(path_ref)):
            wav_path = os.path.join(path_ref, file)
            wav = preprocess_wav(wav_path)

            output = encoder.embed_utterance(wav)
            ref_embds.append(torch.from_numpy(output).to(device))

        for file in tqdm(os.listdir(path_deg)):
            wav_path = os.path.join(path_deg, file)
            wav = preprocess_wav(wav_path)

            output = encoder.embed_utterance(wav)
            deg_embds.append(torch.from_numpy(output).to(device))

    similarity_mode = kwargs["similarity_mode"]
    scores = []

    if similarity_mode == "pairwith":
        for ref_embd, deg_embd in zip(ref_embds, deg_embds):
            scores.append(
                F.cosine_similarity(ref_embd, deg_embd, dim=-1).detach().cpu().numpy()
            )
    elif similarity_mode == "overall":
        for ref_embd in ref_embds:
            for deg_embd in deg_embds:
                scores.append(
                    F.cosine_similarity(ref_embd, deg_embd, dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )

    return np.mean(scores)
