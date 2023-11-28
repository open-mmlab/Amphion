# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import librosa
from tqdm import tqdm
from collections import defaultdict

from utils.util import has_existed
from preprocessors import GOLDEN_TEST_SAMPLES


def get_test_songs():
    golden_samples = GOLDEN_TEST_SAMPLES["m4singer"]
    # every item is a tuple (singer, song)
    golden_songs = [s.split("_")[:2] for s in golden_samples]
    # singer_song, eg: Alto-1_美错
    golden_songs = ["_".join(t) for t in golden_songs]
    return golden_songs


def m4singer_statistics(meta):
    singers = []
    songs = []
    singer2songs = defaultdict(lambda: defaultdict(list))
    for utt in meta:
        p, s, uid = utt["item_name"].split("#")
        singers.append(p)
        songs.append(s)
        singer2songs[p][s].append(uid)

    unique_singers = list(set(singers))
    unique_songs = list(set(songs))
    unique_singers.sort()
    unique_songs.sort()

    print(
        "M4Singer: {} singers, {} utterances ({} unique songs)".format(
            len(unique_singers), len(songs), len(unique_songs)
        )
    )
    print("Singers: \n{}".format("\t".join(unique_singers)))
    return singer2songs, unique_singers


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for m4singer...\n")

    save_dir = os.path.join(output_path, "m4singer")
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
    m4singer_dir = dataset_path
    meta_file = os.path.join(m4singer_dir, "meta.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    singer2songs, unique_singers = m4singer_statistics(meta)

    test_songs = get_test_songs()

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for singer, songs in tqdm(singer2songs.items()):
        song_names = list(songs.keys())

        for chosen_song in song_names:
            chosen_song = chosen_song.replace(" ", "-")
            for chosen_uid in songs[chosen_song]:
                res = {
                    "Dataset": "m4singer",
                    "Singer": singer,
                    "Song": chosen_song,
                    "Uid": "{}_{}_{}".format(singer, chosen_song, chosen_uid),
                }

                res["Path"] = os.path.join(
                    m4singer_dir, "{}#{}/{}.wav".format(singer, chosen_song, chosen_uid)
                )
                assert os.path.exists(res["Path"])

                duration = librosa.get_duration(filename=res["Path"])
                res["Duration"] = duration

                if "_".join([singer, chosen_song]) in test_songs:
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
