# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import os
import json
import librosa
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from utils.util import has_existed
from preprocessors import GOLDEN_TEST_SAMPLES


def get_test_songs():
    golden_samples = GOLDEN_TEST_SAMPLES["opensinger"]
    # every item is a tuple (singer, song)
    golden_songs = [s.split("_")[:3] for s in golden_samples]
    # singer_song, eg: Female1#Almost_lover_Amateur
    return golden_songs


def opensinger_statistics(data_dir):
    singers = []
    songs = []
    singer2songs = defaultdict(lambda: defaultdict(list))

    gender_infos = glob(data_dir + "/*")

    for gender_info in gender_infos:
        gender_info_split = gender_info.split("/")[-1][:-3]

        singer_and_song_infos = glob(gender_info + "/*")

        for singer_and_song_info in singer_and_song_infos:
            singer_and_song_info_split = singer_and_song_info.split("/")[-1].split("_")
            singer_id, song = (
                singer_and_song_info_split[0],
                singer_and_song_info_split[1],
            )
            singer = gender_info_split + "_" + singer_id
            singers.append(singer)
            songs.append(song)

            utts = glob(singer_and_song_info + "/*.wav")

            for utt in utts:
                uid = utt.split("/")[-1].split("_")[-1].split(".")[0]
                singer2songs[singer][song].append(uid)

    unique_singers = list(set(singers))
    unique_songs = list(set(songs))
    unique_singers.sort()
    unique_songs.sort()

    print(
        "opensinger: {} singers, {} songs ({} unique songs)".format(
            len(unique_singers), len(songs), len(unique_songs)
        )
    )
    print("Singers: \n{}".format("\t".join(unique_singers)))
    return singer2songs, unique_singers


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for opensinger...\n")

    save_dir = os.path.join(output_path, "opensinger")
    os.makedirs(save_dir, exist_ok=True)
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    if (
        has_existed(train_output_file)
        and has_existed(test_output_file)
        and has_existed(singer_dict_file)
        and has_existed(utt2singer_file)
    ):
        return
    utt2singer = open(utt2singer_file, "w")

    # Load
    opensinger_path = dataset_path

    singer2songs, unique_singers = opensinger_statistics(opensinger_path)
    test_songs = get_test_songs()

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for i, (singer, songs) in enumerate(singer2songs.items()):
        song_names = list(songs.keys())

        for chosen_song in tqdm(
            song_names, desc="Singer {}/{}".format(i, len(singer2songs))
        ):
            for chosen_uid in songs[chosen_song]:
                res = {
                    "Dataset": "opensinger",
                    "Singer": singer,
                    "Song": chosen_song,
                    "Uid": "{}_{}_{}".format(singer, chosen_song, chosen_uid),
                }
                res["Path"] = "{}Raw/{}_{}/{}_{}_{}.wav".format(
                    singer.split("_")[0],
                    singer.split("_")[1],
                    chosen_song,
                    singer.split("_")[1],
                    chosen_song,
                    chosen_uid,
                )
                res["Path"] = os.path.join(opensinger_path, res["Path"])
                assert os.path.exists(res["Path"])

                duration = librosa.get_duration(filename=res["Path"])
                res["Duration"] = duration

                if duration > 30:
                    print(
                        "Wav file: {}, the duration = {:.2f}s > 30s, which has been abandoned.".format(
                            res["Path"], duration
                        )
                    )
                    continue

                if (
                    [singer.split("_")[0], singer.split("_")[1], chosen_song]
                ) in test_songs:
                    res["index"] = test_index_count
                    test_total_duration += duration
                    test.append(res)
                    test_index_count += 1
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
    singer_lut = {name: i for i, name in enumerate(unique_singers)}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)
