# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from https://huggingface.co/m-a-p/MERT-v1-330M

import torch
from tqdm import tqdm
import numpy as np

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torchaudio
import torchaudio.transforms as T
from sklearn.preprocessing import StandardScaler


def mert_encoder(model, processor, audio_path, hps):
    """
    # mert default sr: 24000
    """
    with torch.no_grad():
        resample_rate = processor.sampling_rate
        device = next(model.parameters()).device

        input_audio, sampling_rate = torchaudio.load(audio_path)
        input_audio = input_audio.squeeze()

        if sampling_rate != resample_rate:
            resampler = T.Resample(sampling_rate, resample_rate)
            input_audio = resampler(input_audio)

        inputs = processor(
            input_audio, sampling_rate=resample_rate, return_tensors="pt"
        ).to(
            device
        )  # {input_values: tensor, attention_mask: tensor}

        outputs = model(**inputs, output_hidden_states=True)  # list: len is 25

    # [25 layer, Time steps, 1024 feature_dim]
    # all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    # mert_features.append(all_layer_hidden_states)

    feature = outputs.hidden_states[
        hps.mert_feature_layer
    ].squeeze()  # [1, frame len, 1024] ->  [frame len, 1024]

    return feature.cpu().detach().numpy()


def mert_features_normalization(raw_mert_features):
    normalized_mert_features = list()

    mert_features = np.array(raw_mert_features)
    scaler = StandardScaler().fit(mert_features)
    for raw_mert_feature in raw_mert_feature:
        normalized_mert_feature = scaler.transform(raw_mert_feature)
        normalized_mert_features.append(normalized_mert_feature)
    return normalized_mert_features


def get_mapped_mert_features(raw_mert_features, mapping_features, fast_mapping=True):
    source_hop = 320
    target_hop = 256

    factor = np.gcd(source_hop, target_hop)
    source_hop //= factor
    target_hop //= factor
    print(
        "Mapping source's {} frames => target's {} frames".format(
            target_hop, source_hop
        )
    )

    mert_features = []
    for index, mapping_feat in enumerate(tqdm(mapping_features)):
        # mapping_feat: (mels_frame_len, n_mels)
        target_len = mapping_feat.shape[0]

        # (frame_len, 1024)
        raw_feats = raw_mert_features[index].cpu().numpy()
        source_len, width = raw_feats.shape

        # const ~= target_len * target_hop
        const = source_len * source_hop // target_hop * target_hop

        # (source_len * source_hop, dim)
        up_sampling_feats = np.repeat(raw_feats, source_hop, axis=0)
        # (const, dim) -> (const/target_hop, target_hop, dim) -> (const/target_hop, dim)
        down_sampling_feats = np.average(
            up_sampling_feats[:const].reshape(-1, target_hop, width), axis=1
        )

        err = abs(target_len - len(down_sampling_feats))
        if err > 3:
            print("index:", index)
            print("mels:", mapping_feat.shape)
            print("raw mert vector:", raw_feats.shape)
            print("up_sampling:", up_sampling_feats.shape)
            print("const:", const)
            print("down_sampling_feats:", down_sampling_feats.shape)
            exit()
        if len(down_sampling_feats) < target_len:
            # (1, dim) -> (err, dim)
            end = down_sampling_feats[-1][None, :].repeat(err, axis=0)
            down_sampling_feats = np.concatenate([down_sampling_feats, end], axis=0)

        # (target_len, dim)
        feats = down_sampling_feats[:target_len]
        mert_features.append(feats)

    return mert_features


def load_mert_model(hps):
    print("Loading MERT Model: ", hps.mert_model)

    # Load model
    model_name = hps.mert_model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    if torch.cuda.is_available():
        model = model.cuda()

    # model = model.eval()

    preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name, trust_remote_code=True
    )
    return model, preprocessor


# loading the corresponding preprocessor config
# def load_preprocessor (model_name="m-a-p/MERT-v1-330M"):
#     print('load_preprocessor...')
#     preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(model_name,trust_remote_code=True)
#     return preprocessor
