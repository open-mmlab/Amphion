# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torchaudio
from tqdm import tqdm
from glob import glob

from utils.util import has_existed


def main(output_path, dataset_path):
    print("-" * 10)
    print("Dataset splits for ljspeech...\n")

    save_dir = os.path.join(output_path, "ljspeech")
    ljspeech_path = dataset_path

    wave_files = glob(ljspeech_path + "/wavs/*.wav")

    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")

    if has_existed(train_output_file):
        return

    utts = []

    for wave_file in tqdm(wave_files):
        res = {
            "Dataset": "ljspeech",
            "Singer": "female1",
            "Uid": "{}".format(wave_file.split("/")[-1].split(".")[0]),
        }
        res["Path"] = wave_file
        assert os.path.exists(res["Path"])

        waveform, sample_rate = torchaudio.load(res["Path"])
        duration = waveform.size(-1) / sample_rate
        res["Duration"] = duration

        if duration <= 1e-8:
            continue

        utts.append(res)

    test_length = len(utts) // 20

    train_utts = []
    train_index_count = 0
    train_total_duration = 0

    for i in tqdm(range(len(utts) - test_length)):
        tmp = utts[i]
        tmp["index"] = train_index_count
        train_index_count += 1
        train_total_duration += tmp["Duration"]
        train_utts.append(tmp)

    test_utts = []
    test_index_count = 0
    test_total_duration = 0

    for i in tqdm(range(len(utts) - test_length, len(utts))):
        tmp = utts[i]
        tmp["index"] = test_index_count
        test_index_count += 1
        test_total_duration += tmp["Duration"]
        test_utts.append(tmp)

    print("#Train = {}, #Test = {}".format(len(train_utts), len(test_utts)))
    print(
        "#Train hours= {}, #Test hours= {}".format(
            train_total_duration / 3600, test_total_duration / 3600
        )
    )

    # Save
    os.makedirs(save_dir, exist_ok=True)
    with open(train_output_file, "w") as f:
        json.dump(train_utts, f, indent=4, ensure_ascii=False)
    with open(test_output_file, "w") as f:
        json.dump(test_utts, f, indent=4, ensure_ascii=False)
