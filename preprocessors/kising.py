# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import os
import json
import torchaudio
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from utils.util import has_existed
from preprocessors import GOLDEN_TEST_SAMPLES


def get_test_folders():
    golden_samples = GOLDEN_TEST_SAMPLES["kising"]
    # every item is a string
    golden_folders = [s.split("_")[:1] for s in golden_samples]
    # folder, eg: 422
    return golden_folders


def KiSing_statistics(data_dir):
    folders = []
    folders2utts = defaultdict(list)

    folder_infos = glob(data_dir + "/*")

    for folder_info in folder_infos:
        folder = folder_info.split("/")[-1]

        folders.append(folder)

        utts = glob(folder_info + "/*.wav")

        for utt in utts:
            uid = utt.split("/")[-1].split(".")[0]
            folders2utts[folder].append(uid)

    unique_folders = list(set(folders))
    unique_folders.sort()

    print("KiSing: {} unique songs".format(len(unique_folders)))
    return folders2utts


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for KiSing...\n")

    save_dir = os.path.join(output_path, "kising")
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    if has_existed(test_output_file):
        return

    # Load
    KiSing_dir = dataset_path

    folders2utts = KiSing_statistics(KiSing_dir)
    test_folders = get_test_folders()

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    folder_names = list(folders2utts.keys())

    for chosen_folder in folder_names:
        for chosen_uid in folders2utts[chosen_folder]:
            res = {
                "Dataset": "kising",
                "Singer": "female1",
                "Uid": "{}_{}".format(chosen_folder, chosen_uid),
            }
            res["Path"] = "{}/{}.wav".format(chosen_folder, chosen_uid)
            res["Path"] = os.path.join(KiSing_dir, res["Path"])
            assert os.path.exists(res["Path"])

            waveform, sample_rate = torchaudio.load(res["Path"])
            duration = waveform.size(-1) / sample_rate
            res["Duration"] = duration

            if ([chosen_folder]) in test_folders:
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
