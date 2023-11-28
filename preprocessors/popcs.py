# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torchaudio
from glob import glob
from collections import defaultdict

from utils.util import has_existed
from preprocessors import GOLDEN_TEST_SAMPLES


def get_test_songs():
    golden_samples = GOLDEN_TEST_SAMPLES["popcs"]
    # every item is a string
    golden_songs = [s.split("_")[:1] for s in golden_samples]
    # song, eg: 万有引力
    return golden_songs


def popcs_statistics(data_dir):
    songs = []
    songs2utts = defaultdict(list)

    song_infos = glob(data_dir + "/*")

    for song_info in song_infos:
        song_info_split = song_info.split("/")[-1].split("-")[-1]

        songs.append(song_info_split)

        utts = glob(song_info + "/*.wav")

        for utt in utts:
            uid = utt.split("/")[-1].split("_")[0]
            songs2utts[song_info_split].append(uid)

    unique_songs = list(set(songs))
    unique_songs.sort()

    print(
        "popcs: {} utterances ({} unique songs)".format(len(songs), len(unique_songs))
    )
    print("Songs: \n{}".format("\t".join(unique_songs)))
    return songs2utts


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for popcs...\n")

    save_dir = os.path.join(output_path, "popcs")
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    if has_existed(test_output_file):
        return

    # Load
    popcs_dir = dataset_path

    songs2utts = popcs_statistics(popcs_dir)
    test_songs = get_test_songs()

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    song_names = list(songs2utts.keys())

    for chosen_song in song_names:
        for chosen_uid in songs2utts[chosen_song]:
            res = {
                "Dataset": "popcs",
                "Singer": "female1",
                "Song": chosen_song,
                "Uid": "{}_{}".format(chosen_song, chosen_uid),
            }
            res["Path"] = "popcs-{}/{}_wf0.wav".format(chosen_song, chosen_uid)
            res["Path"] = os.path.join(popcs_dir, res["Path"])
            assert os.path.exists(res["Path"])

            waveform, sample_rate = torchaudio.load(res["Path"])
            duration = waveform.size(-1) / sample_rate
            res["Duration"] = duration

            if ([chosen_song]) in test_songs:
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
