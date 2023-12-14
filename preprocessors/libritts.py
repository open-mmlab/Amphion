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


def libritts_statistics(data_dir):
    speakers = []
    distribution2speakers2pharases2utts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    distribution_infos = glob(data_dir + "/*")

    for distribution_info in distribution_infos:
        distribution = distribution_info.split("/")[-1]
        print(distribution)

        speaker_infos = glob(distribution_info + "/*")

        if len(speaker_infos) == 0:
            continue

        for speaker_info in speaker_infos:
            speaker = speaker_info.split("/")[-1]

            speakers.append(speaker)

            pharase_infos = glob(speaker_info + "/*")

            for pharase_info in pharase_infos:
                pharase = pharase_info.split("/")[-1]

                utts = glob(pharase_info + "/*.wav")

                for utt in utts:
                    uid = utt.split("/")[-1].split(".")[0]
                    distribution2speakers2pharases2utts[distribution][speaker][
                        pharase
                    ].append(uid)

    unique_speakers = list(set(speakers))
    unique_speakers.sort()

    print("Speakers: \n{}".format("\t".join(unique_speakers)))
    return distribution2speakers2pharases2utts, unique_speakers


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing samples for libritts...\n")

    save_dir = os.path.join(output_path, "libritts")
    os.makedirs(save_dir, exist_ok=True)
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    valid_output_file = os.path.join(save_dir, "valid.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    if has_existed(train_output_file):
        return
    utt2singer = open(utt2singer_file, "w")

    # Load
    libritts_path = dataset_path

    distribution2speakers2pharases2utts, unique_speakers = libritts_statistics(
        libritts_path
    )

    # We select pharases of standard spekaer as test songs
    train = []
    test = []
    valid = []

    train_index_count = 0
    test_index_count = 0
    valid_index_count = 0

    train_total_duration = 0
    test_total_duration = 0
    valid_total_duration = 0

    for distribution, speakers2pharases2utts in tqdm(
        distribution2speakers2pharases2utts.items()
    ):
        for speaker, pharases2utts in tqdm(speakers2pharases2utts.items()):
            pharase_names = list(pharases2utts.keys())

            for chosen_pharase in pharase_names:
                for chosen_uid in pharases2utts[chosen_pharase]:
                    res = {
                        "Dataset": "libritts",
                        "Singer": speaker,
                        "Uid": "{}#{}#{}#{}".format(
                            distribution, speaker, chosen_pharase, chosen_uid
                        ),
                    }
                    res["Path"] = "{}/{}/{}/{}.wav".format(
                        distribution, speaker, chosen_pharase, chosen_uid
                    )
                    res["Path"] = os.path.join(libritts_path, res["Path"])
                    assert os.path.exists(res["Path"])

                    text_file_path = os.path.join(
                        libritts_path,
                        distribution,
                        speaker,
                        chosen_pharase,
                        chosen_uid + ".normalized.txt",
                    )
                    with open(text_file_path, "r") as f:
                        lines = f.readlines()
                        assert len(lines) == 1
                        text = lines[0].strip()
                        res["Text"] = text

                    waveform, sample_rate = torchaudio.load(res["Path"])
                    duration = waveform.size(-1) / sample_rate
                    res["Duration"] = duration

                    if "test" in distribution:
                        res["index"] = test_index_count
                        test_total_duration += duration
                        test.append(res)
                        test_index_count += 1
                    elif "train" in distribution:
                        res["index"] = train_index_count
                        train_total_duration += duration
                        train.append(res)
                        train_index_count += 1
                    elif "dev" in distribution:
                        res["index"] = valid_index_count
                        valid_total_duration += duration
                        valid.append(res)
                        valid_index_count += 1

                    utt2singer.write("{}\t{}\n".format(res["Uid"], res["Singer"]))

    print(
        "#Train = {}, #Test = {}, #Valid = {}".format(len(train), len(test), len(valid))
    )
    print(
        "#Train hours= {}, #Test hours= {}, #Valid hours= {}".format(
            train_total_duration / 3600,
            test_total_duration / 3600,
            valid_total_duration / 3600,
        )
    )

    # Save train.json and test.json
    with open(train_output_file, "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(test_output_file, "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
    with open(valid_output_file, "w") as f:
        json.dump(valid, f, indent=4, ensure_ascii=False)

    # Save singers.json
    singer_lut = {name: i for i, name in enumerate(unique_speakers)}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)
