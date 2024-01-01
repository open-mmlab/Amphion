# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from resemblyzer import VoiceEncoder, preprocess_wav


def load_wavs(directory):
    """Load all WAV files from the given directory.

    Args:
        directory (str): Path to the directory containing WAV files.
    """

    wavs = []
    wav_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            path = os.path.join(directory, filename)
            wav = preprocess_wav(path)
            wavs.append(wav)
            wav_names.append(filename)
    return wavs, wav_names


def calculate_cosine_similarity(embeddings1, embeddings2, names1, names2):
    """Calculate cosine similarity between two sets of embeddings.

    Args:
        embeddings1 & 2 (list): List of embeddings.
        names1 & 2 (list): List of filenames.
    """

    similarity_info = []
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            emb1_tensor = torch.tensor(emb1).unsqueeze(0)
            emb2_tensor = torch.tensor(emb2).unsqueeze(0)
            similarity = F.cosine_similarity(emb1_tensor, emb2_tensor)
            similarity_info.append(
                {"Reference": names2[j], "Target": names1[i], "Similarity": similarity}
            )
    return similarity_info


def extract_resemblyzer_similarity(target_path, reference_path, dump_dir):
    """Extract similarity between utterances using resemblyzer.

    Args:
        target_path (str): Path to the directory containing target utterances.
        reference_path (str): Path to the directory containing reference utterances.
        dump_dir (str): Path to the directory where similarity results are dumped.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = target_path.split("/")[-1]
    csv_file_name = f"similarity_results_{filename}.csv"
    dump_dir = dump_dir + "/" + csv_file_name

    target_wavs, target_names = load_wavs(target_path)
    ref_wavs, ref_names = load_wavs(reference_path)

    encoder = VoiceEncoder().to(device)

    target_embds = [encoder.embed_utterance(wav) for wav in target_wavs]
    ref_embds = [encoder.embed_utterance(wav) for wav in ref_wavs]

    similarity_info = calculate_cosine_similarity(
        target_embds, ref_embds, target_names, ref_names
    )

    df = pd.DataFrame(similarity_info)
    average_similarity_per_utterance = (
        df.groupby("Target")["Similarity"].mean().reset_index()
    )
    average_similarity_per_utterance.to_csv(dump_dir, index=False)
    average_similarity_per_utterance.head()
    avg_overall = average_similarity_per_utterance["Similarity"].mean()
    print("Overall similarity: ", avg_overall)
    print(f"Per utterance similarity results are saved in {dump_dir}")

    return avg_overall
