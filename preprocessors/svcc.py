# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import librosa
import json

from utils.util import has_existed
from preprocessors import GOLDEN_TEST_SAMPLES


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing training dataset for svcc...")

    data_dir = os.path.join(dataset_path, "Data")
    save_dir = os.path.join(output_path, "svcc")
    os.makedirs(save_dir, exist_ok=True)

    singer_dict_file = os.path.join(save_dir, "singers.json")
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    utt2singer = open(utt2singer_file, "w")

    # Load utterances
    train = []
    test = []
    singers = []

    for wav_file in glob.glob(os.path.join(data_dir, "*/*.wav")):
        singer, filename = wav_file.split("/")[-2:]
        uid = filename.split(".")[0]
        utt = {
            "Dataset": "svcc",
            "Singer": singer,
            "Uid": "{}_{}".format(singer, uid),
            "Path": wav_file,
        }

        # Duration
        duration = librosa.get_duration(filename=wav_file)
        utt["Duration"] = duration

        if utt["Uid"] in GOLDEN_TEST_SAMPLES["svcc"]:
            test.append(utt)
        else:
            train.append(utt)

        singers.append(singer)
        utt2singer.write("{}\t{}\n".format(utt["Uid"], utt["Singer"]))

    # Save singers.json
    unique_singers = list(set(singers))
    unique_singers.sort()
    singer_lut = {name: i for i, name in enumerate(unique_singers)}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)

    train_total_duration = sum([utt["Duration"] for utt in train])
    test_total_duration = sum([utt["Duration"] for utt in test])

    for dataset_type in ["train", "test"]:
        output_file = os.path.join(save_dir, "{}.json".format(dataset_type))
        if has_existed(output_file):
            continue

        utterances = eval(dataset_type)
        utterances = sorted(utterances, key=lambda x: x["Uid"])

        for i in range(len(utterances)):
            utterances[i]["index"] = i

        print("{}: Total size: {}\n".format(dataset_type, len(utterances)))

        # Save
        with open(output_file, "w") as f:
            json.dump(utterances, f, indent=4, ensure_ascii=False)

    print(
        "#Train hours= {}, #Test hours= {}".format(
            train_total_duration / 3600, test_total_duration / 3600
        )
    )
