# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def intersperse(lst, item):
    """
    Insert an item in between any two consecutive elements of the given list, including beginning and end of list

    Example:
        >>> intersperse(0, [1, 74, 5, 31])
            [0, 1, 0, 74, 0, 5, 0, 31, 0]
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def load_content_feature_path(meta_data, processed_dir, feat_dir):
    utt2feat_path = {}
    for utt_info in meta_data:
        utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
        feat_path = os.path.join(
            processed_dir, utt_info["Dataset"], feat_dir, f'{utt_info["Uid"]}.npy'
        )
        utt2feat_path[utt] = feat_path

    return utt2feat_path


def load_source_content_feature_path(meta_data, feat_dir):
    utt2feat_path = {}
    for utt in meta_data:
        feat_path = os.path.join(feat_dir, f"{utt}.npy")
        utt2feat_path[utt] = feat_path

    return utt2feat_path


def get_spk_map(spk2id_path, utt2spk_path):
    utt2spk = {}
    with open(spk2id_path, "r") as spk2id_file:
        spk2id = json.load(spk2id_file)
    with open(utt2spk_path, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk = line.strip().split("\t")
            utt2spk[utt] = spk
    return spk2id, utt2spk


def get_target_f0_median(f0_dir):
    total_f0 = []
    for utt in os.listdir(f0_dir):
        if not utt.endswith(".npy"):
            continue
        f0_feat_path = os.path.join(f0_dir, utt)
        f0 = np.load(f0_feat_path)
        total_f0 += f0.tolist()

    total_f0 = np.array(total_f0)
    voiced_position = np.where(total_f0 != 0)
    return np.median(total_f0[voiced_position])


def get_conversion_f0_factor(source_f0, target_median, source_median=None):
    """Align the median between source f0 and target f0

    Note: Here we use multiplication, whose factor is target_median/source_median

    Reference: Frequency and pitch interval
    http://blog.ccyg.studio/article/be12c2ee-d47c-4098-9782-ca76da3035e4/
    """
    if source_median is None:
        voiced_position = np.where(source_f0 != 0)
        source_median = np.median(source_f0[voiced_position])
    factor = target_median / source_median
    return source_median, factor


def transpose_key(frame_pitch, trans_key):
    # Transpose by user's argument
    print("Transpose key = {} ...\n".format(trans_key))

    transed_pitch = frame_pitch * 2 ** (trans_key / 12)
    return transed_pitch


def pitch_shift_to_target(frame_pitch, target_pitch_median, source_pitch_median=None):
    # Loading F0 Base (median) and shift
    source_pitch_median, factor = get_conversion_f0_factor(
        frame_pitch, target_pitch_median, source_pitch_median
    )
    print(
        "Auto transposing: source f0 median = {:.1f}, target f0 median = {:.1f}, factor = {:.2f}".format(
            source_pitch_median, target_pitch_median, factor
        )
    )
    transed_pitch = frame_pitch * factor
    return transed_pitch


def load_frame_pitch(
    meta_data,
    processed_dir,
    pitch_dir,
    use_log_scale=False,
    return_norm=False,
    interoperate=False,
    utt2spk=None,
):
    utt2pitch = {}
    utt2uv = {}
    if utt2spk is None:
        pitch_scaler = StandardScaler()
        for utt_info in meta_data:
            utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
            pitch_path = os.path.join(
                processed_dir, utt_info["Dataset"], pitch_dir, f'{utt_info["Uid"]}.npy'
            )
            pitch = np.load(pitch_path)
            assert len(pitch) > 0
            uv = pitch != 0
            utt2uv[utt] = uv
            if use_log_scale:
                nonzero_idxes = np.where(pitch != 0)[0]
                pitch[nonzero_idxes] = np.log(pitch[nonzero_idxes])
            utt2pitch[utt] = pitch
            pitch_scaler.partial_fit(pitch.reshape(-1, 1))

        mean, std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        if return_norm:
            for utt_info in meta_data:
                utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
                pitch = utt2pitch[utt]
                normalized_pitch = (pitch - mean) / std
                utt2pitch[utt] = normalized_pitch
        pitch_statistic = {"mean": mean, "std": std}
    else:
        spk2utt = {}
        pitch_statistic = []
        for utt_info in meta_data:
            utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
            if not utt2spk[utt] in spk2utt:
                spk2utt[utt2spk[utt]] = []
            spk2utt[utt2spk[utt]].append(utt)

        for spk in spk2utt:
            pitch_scaler = StandardScaler()
            for utt in spk2utt[spk]:
                dataset = utt.split("_")[0]
                uid = "_".join(utt.split("_")[1:])
                pitch_path = os.path.join(
                    processed_dir, dataset, pitch_dir, f"{uid}.npy"
                )
                pitch = np.load(pitch_path)
                assert len(pitch) > 0
                uv = pitch != 0
                utt2uv[utt] = uv
                if use_log_scale:
                    nonzero_idxes = np.where(pitch != 0)[0]
                    pitch[nonzero_idxes] = np.log(pitch[nonzero_idxes])
                utt2pitch[utt] = pitch
                pitch_scaler.partial_fit(pitch.reshape(-1, 1))

            mean, std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
            if return_norm:
                for utt in spk2utt[spk]:
                    pitch = utt2pitch[utt]
                    normalized_pitch = (pitch - mean) / std
                    utt2pitch[utt] = normalized_pitch
            pitch_statistic.append({"spk": spk, "mean": mean, "std": std})

    return utt2pitch, utt2uv, pitch_statistic


# discard
def load_phone_pitch(
    meta_data,
    processed_dir,
    pitch_dir,
    utt2dur,
    use_log_scale=False,
    return_norm=False,
    interoperate=True,
    utt2spk=None,
):
    print("Load Phone Pitch")
    utt2pitch = {}
    utt2uv = {}
    if utt2spk is None:
        pitch_scaler = StandardScaler()
        for utt_info in tqdm(meta_data):
            utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
            pitch_path = os.path.join(
                processed_dir, utt_info["Dataset"], pitch_dir, f'{utt_info["Uid"]}.npy'
            )
            frame_pitch = np.load(pitch_path)
            assert len(frame_pitch) > 0
            uv = frame_pitch != 0
            utt2uv[utt] = uv
            phone_pitch = phone_average_pitch(frame_pitch, utt2dur[utt], interoperate)
            if use_log_scale:
                nonzero_idxes = np.where(phone_pitch != 0)[0]
                phone_pitch[nonzero_idxes] = np.log(phone_pitch[nonzero_idxes])
            utt2pitch[utt] = phone_pitch
            pitch_scaler.partial_fit(remove_outlier(phone_pitch).reshape(-1, 1))

        mean, std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        if return_norm:
            for utt_info in meta_data:
                utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
                pitch = utt2pitch[utt]
                normalized_pitch = (pitch - mean) / std
                max_value = max(max_value, max(normalized_pitch))
                min_value = min(min_value, min(normalized_pitch))
                utt2pitch[utt] = normalized_pitch
                phone_normalized_pitch_path = os.path.join(
                    processed_dir,
                    utt_info["Dataset"],
                    "phone_level_" + pitch_dir,
                    f'{utt_info["Uid"]}.npy',
                )
        pitch_statistic = {
            "mean": mean,
            "std": std,
            "min_value": min_value,
            "max_value": max_value,
        }
    else:
        spk2utt = {}
        pitch_statistic = []
        for utt_info in tqdm(meta_data):
            utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
            if not utt2spk[utt] in spk2utt:
                spk2utt[utt2spk[utt]] = []
            spk2utt[utt2spk[utt]].append(utt)

        for spk in spk2utt:
            pitch_scaler = StandardScaler()
            for utt in spk2utt[spk]:
                dataset = utt.split("_")[0]
                uid = "_".join(utt.split("_")[1:])
                pitch_path = os.path.join(
                    processed_dir, dataset, pitch_dir, f"{uid}.npy"
                )
                frame_pitch = np.load(pitch_path)
                assert len(frame_pitch) > 0
                uv = frame_pitch != 0
                utt2uv[utt] = uv
                phone_pitch = phone_average_pitch(
                    frame_pitch, utt2dur[utt], interoperate
                )
                if use_log_scale:
                    nonzero_idxes = np.where(phone_pitch != 0)[0]
                    phone_pitch[nonzero_idxes] = np.log(phone_pitch[nonzero_idxes])
                utt2pitch[utt] = phone_pitch
                pitch_scaler.partial_fit(remove_outlier(phone_pitch).reshape(-1, 1))

            mean, std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
            max_value = np.finfo(np.float64).min
            min_value = np.finfo(np.float64).max

            if return_norm:
                for utt in spk2utt[spk]:
                    pitch = utt2pitch[utt]
                    normalized_pitch = (pitch - mean) / std
                    max_value = max(max_value, max(normalized_pitch))
                    min_value = min(min_value, min(normalized_pitch))
                    utt2pitch[utt] = normalized_pitch
            pitch_statistic.append(
                {
                    "spk": spk,
                    "mean": mean,
                    "std": std,
                    "min_value": min_value,
                    "max_value": max_value,
                }
            )

    return utt2pitch, utt2uv, pitch_statistic


def phone_average_pitch(pitch, dur, interoperate=False):
    pos = 0

    if interoperate:
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))
    phone_pitch = np.zeros(len(dur))

    for i, d in enumerate(dur):
        d = int(d)
        if d > 0 and pos < len(pitch):
            phone_pitch[i] = np.mean(pitch[pos : pos + d])
        else:
            phone_pitch[i] = 0
        pos += d
    return phone_pitch


def load_energy(
    meta_data,
    processed_dir,
    energy_dir,
    use_log_scale=False,
    return_norm=False,
    utt2spk=None,
):
    utt2energy = {}
    if utt2spk is None:
        for utt_info in meta_data:
            utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
            energy_path = os.path.join(
                processed_dir, utt_info["Dataset"], energy_dir, f'{utt_info["Uid"]}.npy'
            )
            if not os.path.exists(energy_path):
                continue
            energy = np.load(energy_path)
            assert len(energy) > 0

            if use_log_scale:
                nonzero_idxes = np.where(energy != 0)[0]
                energy[nonzero_idxes] = np.log(energy[nonzero_idxes])
            utt2energy[utt] = energy

        if return_norm:
            with open(
                os.path.join(
                    processed_dir, utt_info["Dataset"], energy_dir, "statistics.json"
                )
            ) as f:
                stats = json.load(f)
                mean, std = (
                    stats[utt_info["Dataset"] + "_" + utt_info["Singer"]][
                        "voiced_positions"
                    ]["mean"],
                    stats["LJSpeech_LJSpeech"]["voiced_positions"]["std"],
                )
            for utt in utt2energy.keys():
                energy = utt2energy[utt]
                normalized_energy = (energy - mean) / std
                utt2energy[utt] = normalized_energy

        energy_statistic = {"mean": mean, "std": std}
    else:
        spk2utt = {}
        energy_statistic = []
        for utt_info in meta_data:
            utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
            if not utt2spk[utt] in spk2utt:
                spk2utt[utt2spk[utt]] = []
            spk2utt[utt2spk[utt]].append(utt)

        for spk in spk2utt:
            energy_scaler = StandardScaler()
            for utt in spk2utt[spk]:
                dataset = utt.split("_")[0]
                uid = "_".join(utt.split("_")[1:])
                energy_path = os.path.join(
                    processed_dir, dataset, energy_dir, f"{uid}.npy"
                )
                if not os.path.exists(energy_path):
                    continue
                frame_energy = np.load(energy_path)
                assert len(frame_energy) > 0

                if use_log_scale:
                    nonzero_idxes = np.where(frame_energy != 0)[0]
                    frame_energy[nonzero_idxes] = np.log(frame_energy[nonzero_idxes])
                utt2energy[utt] = frame_energy
                energy_scaler.partial_fit(frame_energy.reshape(-1, 1))

            mean, std = energy_scaler.mean_[0], energy_scaler.scale_[0]
            if return_norm:
                for utt in spk2utt[spk]:
                    energy = utt2energy[utt]
                    normalized_energy = (energy - mean) / std
                    utt2energy[utt] = normalized_energy
            energy_statistic.append({"spk": spk, "mean": mean, "std": std})

    return utt2energy, energy_statistic


def load_frame_energy(
    meta_data,
    processed_dir,
    energy_dir,
    use_log_scale=False,
    return_norm=False,
    interoperate=False,
    utt2spk=None,
):
    utt2energy = {}
    if utt2spk is None:
        energy_scaler = StandardScaler()
        for utt_info in meta_data:
            utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
            energy_path = os.path.join(
                processed_dir, utt_info["Dataset"], energy_dir, f'{utt_info["Uid"]}.npy'
            )
            frame_energy = np.load(energy_path)
            assert len(frame_energy) > 0

            if use_log_scale:
                nonzero_idxes = np.where(frame_energy != 0)[0]
                frame_energy[nonzero_idxes] = np.log(frame_energy[nonzero_idxes])
            utt2energy[utt] = frame_energy
            energy_scaler.partial_fit(frame_energy.reshape(-1, 1))

        mean, std = energy_scaler.mean_[0], energy_scaler.scale_[0]
        if return_norm:
            for utt_info in meta_data:
                utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
                energy = utt2energy[utt]
                normalized_energy = (energy - mean) / std
                utt2energy[utt] = normalized_energy
        energy_statistic = {"mean": mean, "std": std}

    else:
        spk2utt = {}
        energy_statistic = []
        for utt_info in meta_data:
            utt = utt_info["Dataset"] + "_" + utt_info["Uid"]
            if not utt2spk[utt] in spk2utt:
                spk2utt[utt2spk[utt]] = []
            spk2utt[utt2spk[utt]].append(utt)

        for spk in spk2utt:
            energy_scaler = StandardScaler()
            for utt in spk2utt[spk]:
                dataset = utt.split("_")[0]
                uid = "_".join(utt.split("_")[1:])
                energy_path = os.path.join(
                    processed_dir, dataset, energy_dir, f"{uid}.npy"
                )
                frame_energy = np.load(energy_path)
                assert len(frame_energy) > 0

                if use_log_scale:
                    nonzero_idxes = np.where(frame_energy != 0)[0]
                    frame_energy[nonzero_idxes] = np.log(frame_energy[nonzero_idxes])
                utt2energy[utt] = frame_energy
                energy_scaler.partial_fit(frame_energy.reshape(-1, 1))

            mean, std = energy_scaler.mean_[0], energy_scaler.scale_[0]
            if return_norm:
                for utt in spk2utt[spk]:
                    energy = utt2energy[utt]
                    normalized_energy = (energy - mean) / std
                    utt2energy[utt] = normalized_energy
            energy_statistic.append({"spk": spk, "mean": mean, "std": std})

    return utt2energy, energy_statistic


def align_length(feature, target_len, pad_value=0.0):
    feature_len = feature.shape[-1]
    dim = len(feature.shape)
    # align 1-D data
    if dim == 2:
        if target_len > feature_len:
            feature = np.pad(
                feature,
                ((0, 0), (0, target_len - feature_len)),
                constant_values=pad_value,
            )
        else:
            feature = feature[:, :target_len]
    # align 2-D data
    elif dim == 1:
        if target_len > feature_len:
            feature = np.pad(
                feature, (0, target_len - feature_len), constant_values=pad_value
            )
        else:
            feature = feature[:target_len]
    else:
        raise NotImplementedError
    return feature


def align_whisper_feauture_length(
    feature, target_len, fast_mapping=True, source_hop=320, target_hop=256
):
    factor = np.gcd(source_hop, target_hop)
    source_hop //= factor
    target_hop //= factor
    # print(
    #     "Mapping source's {} frames => target's {} frames".format(
    #         target_hop, source_hop
    #     )
    # )

    max_source_len = 1500
    target_len = min(target_len, max_source_len * source_hop // target_hop)

    width = feature.shape[-1]

    if fast_mapping:
        source_len = target_len * target_hop // source_hop + 1
        feature = feature[:source_len]

    else:
        source_len = max_source_len

    # const ~= target_len * target_hop
    const = source_len * source_hop // target_hop * target_hop

    # (source_len * source_hop, dim)
    up_sampling_feats = np.repeat(feature, source_hop, axis=0)
    # (const, dim) -> (const/target_hop, target_hop, dim) -> (const/target_hop, dim)
    down_sampling_feats = np.average(
        up_sampling_feats[:const].reshape(-1, target_hop, width), axis=1
    )
    assert len(down_sampling_feats) >= target_len

    # (target_len, dim)
    feat = down_sampling_feats[:target_len]

    return feat


def align_content_feature_length(feature, target_len, source_hop=320, target_hop=256):
    factor = np.gcd(source_hop, target_hop)
    source_hop //= factor
    target_hop //= factor
    # print(
    #     "Mapping source's {} frames => target's {} frames".format(
    #         target_hop, source_hop
    #     )
    # )

    # (source_len, 256)
    source_len, width = feature.shape

    # const ~= target_len * target_hop
    const = source_len * source_hop // target_hop * target_hop

    # (source_len * source_hop, dim)
    up_sampling_feats = np.repeat(feature, source_hop, axis=0)
    # (const, dim) -> (const/target_hop, target_hop, dim) -> (const/target_hop, dim)
    down_sampling_feats = np.average(
        up_sampling_feats[:const].reshape(-1, target_hop, width), axis=1
    )

    err = abs(target_len - len(down_sampling_feats))
    if err > 4:  ## why 4 not 3?
        print("target_len:", target_len)
        print("raw feature:", feature.shape)
        print("up_sampling:", up_sampling_feats.shape)
        print("down_sampling_feats:", down_sampling_feats.shape)
        exit()
    if len(down_sampling_feats) < target_len:
        # (1, dim) -> (err, dim)
        end = down_sampling_feats[-1][None, :].repeat(err, axis=0)
        down_sampling_feats = np.concatenate([down_sampling_feats, end], axis=0)

    # (target_len, dim)
    feat = down_sampling_feats[:target_len]

    return feat


def remove_outlier(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)
    return values[normal_indices]
