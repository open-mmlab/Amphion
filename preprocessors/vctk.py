# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import librosa
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from utils.util import has_existed


def get_lines(file):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines


def vctk_statistics(data_dir):
    speakers = []
    speakers2utts = defaultdict(list)

    speaker_infos = glob(data_dir + "/wav48_silence_trimmed" + "/*")

    for speaker_info in speaker_infos:
        speaker = speaker_info.split("/")[-1]

        if speaker == "log.txt":
            continue

        speakers.append(speaker)

        utts = glob(speaker_info + "/*")

        for utt in utts:
            uid = (
                utt.split("/")[-1].split("_")[1]
                + "_"
                + utt.split("/")[-1].split("_")[2].split(".")[0]
            )
            speakers2utts[speaker].append(uid)

    unique_speakers = list(set(speakers))
    unique_speakers.sort()

    print("Speakers: \n{}".format("\t".join(unique_speakers)))
    return speakers2utts, unique_speakers


def vctk_speaker_infos(data_dir):
    file = os.path.join(data_dir, "speaker-info.txt")
    lines = get_lines(file)

    ID2speakers = defaultdict()
    for l in tqdm(lines):
        items = l.replace(" ", "")

        if items[:2] == "ID":
            # The header line
            continue

        if items[0] == "p":
            id = items[:4]
            gender = items[6]
        elif items[0] == "s":
            id = items[:2]
            gender = items[4]

        if gender == "F":
            speaker = "female_{}".format(id)
        elif gender == "M":
            speaker = "male_{}".format(id)

        ID2speakers[id] = speaker

    return ID2speakers


def main(output_path, dataset_path, TEST_NUM_OF_EVERY_SPEAKER=3):
    print("-" * 10)
    print("Preparing test samples for vctk...")

    save_dir = os.path.join(output_path, "vctk")
    os.makedirs(save_dir, exist_ok=True)
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    if has_existed(train_output_file):
        return
    utt2singer = open(utt2singer_file, "w")

    # Load
    vctk_dir = dataset_path

    ID2speakers = vctk_speaker_infos(vctk_dir)
    speaker2utts, unique_speakers = vctk_statistics(vctk_dir)

    # We select speakers of standard samples as test utts
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0
    test_speaker_count = defaultdict(int)

    train_total_duration = 0
    test_total_duration = 0

    for i, speaker in enumerate(speaker2utts.keys()):
        for chosen_uid in tqdm(
            speaker2utts[speaker],
            desc="Speaker {}/{}, #Train = {}, #Test = {}".format(
                i + 1, len(speaker2utts), train_index_count, test_index_count
            ),
        ):
            res = {
                "Dataset": "vctk",
                "Singer": ID2speakers[speaker],
                "Uid": "{}#{}".format(ID2speakers[speaker], chosen_uid),
            }
            res["Path"] = "{}/{}_{}.flac".format(speaker, speaker, chosen_uid)
            res["Path"] = os.path.join(vctk_dir, "wav48_silence_trimmed", res["Path"])
            assert os.path.exists(res["Path"])

            duration = librosa.get_duration(filename=res["Path"])
            res["Duration"] = duration

            if test_speaker_count[speaker] < TEST_NUM_OF_EVERY_SPEAKER:
                res["index"] = test_index_count
                test_total_duration += duration
                test.append(res)
                test_index_count += 1
                test_speaker_count[speaker] += 1
            else:
                res["index"] = train_index_count
                train_total_duration += duration
                train.append(res)
                train_index_count += 1

            utt2singer.write("{}\t{}\n".format(res["Uid"], res["Singer"]))

    print("#Train = {}, #Test = {}".format(len(train), len(test)))
    print(
        "#Train hours= {}, #Test hours= {}".format(
            train_total_duration / 3600, test_total_duration / 3600
        )
    )

    # Save train.json and test.json
    with open(train_output_file, "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(test_output_file, "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)

    # Save singers.json
    singer_lut = {name: i for i, name in enumerate(unique_speakers)}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)
