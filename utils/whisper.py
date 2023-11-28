# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import pickle
from tqdm import tqdm
import numpy as np

from modules import whisper_extractor as whisper


def whisper_encoder_batch(model, audio_paths):
    batch = len(audio_paths)
    batch_mel = torch.zeros((batch, 80, 3000), dtype=torch.float32, device=model.device)

    for i, audio_path in enumerate(audio_paths):
        # (48000,)
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        # (80, 3000)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        batch_mel[i] = mel

    with torch.no_grad():
        # (batch, 1500, 1024)
        features = model.embed_audio(batch_mel)

    return features.cpu().detach().numpy()


def whisper_encoder(model, audio_path):
    audio = whisper.load_audio(str(audio_path))
    audio = whisper.pad_or_trim(audio)

    # (80, 3000)
    mel = whisper.log_mel_spectrogram(audio).to(model.device).unsqueeze(0)

    with torch.no_grad():
        # (1, 1500, 1024) -> # (1500, 1024)
        features = model.embed_audio(mel).squeeze(0)

    return features.cpu().detach().numpy()


def get_mapped_whisper_features(
    raw_whisper_features, mapping_features, fast_mapping=True
):
    """
    Whisper: frameshift = 20ms (30s audio -> 1500 frames), hop_size = 480 in 24k
    # Ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/model.py#L136

    Now it's only used for mapping to bigvgan's mels (sr = 24k, hop_size = 256, frameshift ~= 10.7 ms)
    """
    source_hop = 480
    target_hop = 256

    factor = np.gcd(source_hop, target_hop)
    source_hop //= factor
    target_hop //= factor
    print(
        "Mapping source's {} frames => target's {} frames".format(
            target_hop, source_hop
        )
    )

    max_source_len = 1500
    whisper_features = []
    for index, mapping_feat in enumerate(tqdm(mapping_features)):
        # mapping_feat: (mels_frame_len, n_mels)
        target_len = mapping_feat.shape[0]
        # The max target_len is 2812
        target_len = min(target_len, max_source_len * source_hop // target_hop)

        # (1500, dim)
        raw_feats = raw_whisper_features[index]
        width = raw_feats.shape[-1]

        if fast_mapping:
            source_len = target_len * target_hop // source_hop + 1
            raw_feats = raw_feats[:source_len]
        else:
            source_len = max_source_len

        # const ~= target_len * target_hop
        const = source_len * source_hop // target_hop * target_hop

        # (source_len * source_hop, dim)
        up_sampling_feats = np.repeat(raw_feats, source_hop, axis=0)
        # (const, dim) -> (const/target_hop, target_hop, dim) -> (const/target_hop, dim)
        down_sampling_feats = np.average(
            up_sampling_feats[:const].reshape(-1, target_hop, width), axis=1
        )
        assert len(down_sampling_feats) >= target_len

        # (target_len, dim)
        feats = down_sampling_feats[:target_len]
        whisper_features.append(feats)

    return whisper_features


def load_whisper_model(hps):
    print("Loading Whisper Model: ", hps.whisper_model)
    model = whisper.load_model(hps.whisper_model)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    return model


def load_target_acoustic_features(
    output_path, dataset, acoustic_features_name, acoustic_features_fs, dataset_type
):
    mapping_dir = os.path.join(
        output_path,
        dataset,
        "{}/{}".format(acoustic_features_name, acoustic_features_fs),
    )
    with open(os.path.join(mapping_dir, "{}.pkl".format(dataset_type)), "rb") as f:
        mapping_features = pickle.load(f)

    # Mels: (n_mels, frame_len) -> (frame_len, n_mels)
    if acoustic_features_name == "mels":
        print("Transposing mel features...")
        mapping_features = [feat.T for feat in mapping_features]

    print(
        "Mapping to the acoustic features {}, #sz = {}, feats[0] is {}".format(
            acoustic_features_name, len(mapping_features), mapping_features[0].shape
        )
    )
    return mapping_features


def extract_whisper_features_of_dataset(
    datasets,
    model,
    batch_size,
    out_dir,
):
    audio_paths = [utt["Path"] for utt in datasets]
    if len(audio_paths) < batch_size:
        batch_size = len(audio_paths)

    start, end = 0, 0
    while end < len(audio_paths):
        # Raw features: (batch_size, 1500, dim)
        start = end
        end = start + batch_size
        tmp_raw_whisper_features = whisper_encoder_batch(model, audio_paths[start:end])

        # Mapping to acoustic features' lengths
        for index, utt in enumerate(tqdm(datasets[start:end])):
            uid = utt["Uid"]
            raw_whisper_feature = tmp_raw_whisper_features[index]

            save_path = os.path.join(out_dir, uid + ".npy")
            np.save(save_path, raw_whisper_feature)

        print("{}/{} Done...".format(end, len(audio_paths)))
