# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import os
import librosa
import numpy as np
from tqdm import tqdm

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


def extract_wavlm_similarity(target_path, reference_path):
    """Extract cosine similarity based on WavLM for two given audio folders.
    target_path: path to the ground truth audio folder.
    reference_path: path to the predicted audio folder.
    """
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-base-plus-sv"
    )
    gpu = False
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True

    similarity_scores = []

    for file in tqdm(os.listdir(reference_path)):
        ref_wav_path = os.path.join(reference_path, file)
        tgt_wav_path = os.path.join(target_path, file)

        ref_wav, _ = librosa.load(ref_wav_path, sr=16000)
        tgt_wav, _ = librosa.load(tgt_wav_path, sr=16000)

        inputs = feature_extractor(
            [tgt_wav, ref_wav], padding=True, return_tensors="pt"
        )

        if gpu:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda("cuda")

        with torch.no_grad():
            embeddings = model(**inputs).embeddings
            embeddings = embeddings.cpu()
            cos_sim_score = F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)
            similarity_scores.append(cos_sim_score.item())

    return np.mean(similarity_scores)
