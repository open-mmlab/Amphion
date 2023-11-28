# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import os
import librosa

from utils.util import has_existed


def get_lines(file):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines


def get_uid2utt(opencpop_path, dataset, dataset_type):
    index_count = 0
    total_duration = 0

    file = os.path.join(opencpop_path, "segments", "{}.txt".format(dataset_type))
    lines = get_lines(file)

    uid2utt = []
    for l in tqdm(lines):
        items = l.split("|")
        uid = items[0]

        res = {
            "Dataset": dataset,
            "index": index_count,
            "Singer": "female1",
            "Uid": uid,
        }

        # Duration in wav files
        audio_file = os.path.join(opencpop_path, "segments/wavs/{}.wav".format(uid))
        res["Path"] = audio_file

        duration = librosa.get_duration(filename=res["Path"])
        res["Duration"] = duration

        uid2utt.append(res)

        index_count = index_count + 1
        total_duration += duration

    return uid2utt, total_duration / 3600


def main(dataset, output_path, dataset_path):
    print("-" * 10)
    print("Dataset splits for {}...\n".format(dataset))

    save_dir = os.path.join(output_path, dataset)
    opencpop_path = dataset_path
    for dataset_type in ["train", "test"]:
        output_file = os.path.join(save_dir, "{}.json".format(dataset_type))
        if has_existed(output_file):
            continue

        res, hours = get_uid2utt(opencpop_path, dataset, dataset_type)

        # Save
        os.makedirs(save_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)

        print("{}_{}_hours= {}".format(dataset, dataset_type, hours))
