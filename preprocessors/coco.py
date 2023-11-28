# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torchaudio
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from utils.util import has_existed
from preprocessors import GOLDEN_TEST_SAMPLES


def get_test_songs():
    return ["007Di Da Di"]


def coco_statistics(data_dir):
    song2utts = defaultdict(list)

    song_infos = glob(data_dir + "/*")

    for song in song_infos:
        song_name = song.split("/")[-1]
        utts = glob(song + "/*.wav")
        for utt in utts:
            uid = utt.split("/")[-1].split(".")[0]
            song2utts[song_name].append(uid)

    print("Coco: {} songs".format(len(song_infos)))
    return song2utts


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing datasets for Coco...\n")

    save_dir = os.path.join(output_path, "coco")
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    if has_existed(test_output_file):
        return

    # Load
    song2utts = coco_statistics(dataset_path)
    test_songs = get_test_songs()

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for song_name, uids in tqdm(song2utts.items()):
        for chosen_uid in uids:
            res = {
                "Dataset": "coco",
                "Singer": "coco",
                "Song": song_name,
                "Uid": "{}_{}".format(song_name, chosen_uid),
            }
            res["Path"] = "{}/{}.wav".format(song_name, chosen_uid)
            res["Path"] = os.path.join(dataset_path, res["Path"])
            assert os.path.exists(res["Path"])

            waveform, sample_rate = torchaudio.load(res["Path"])
            duration = waveform.size(-1) / sample_rate
            res["Duration"] = duration

            if song_name in test_songs:
                res["index"] = test_index_count
                test_total_duration += duration
                test.append(res)
                test_index_count += 1
            else:
                res["index"] = train_index_count
                train_total_duration += duration
                train.append(res)
                train_index_count += 1

    print("#Train = {}, #Test = {}".format(len(train), len(test)))
    print(
        "#Train hours= {}, #Test hours= {}".format(
            train_total_duration / 3600, test_total_duration / 3600
        )
    )

    # Save
    os.makedirs(save_dir, exist_ok=True)
    with open(train_output_file, "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(test_output_file, "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
