# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torchaudio
import librosa
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from utils.util import has_existed
from preprocessors import GOLDEN_TEST_SAMPLES


def get_test_songs():
    golden_samples = GOLDEN_TEST_SAMPLES["popbutfy"]
    # every item is a tuple (singer, song)
    golden_songs = [s.split("#")[:2] for s in golden_samples]
    # singer#song, eg: Female1#Almost_lover_Amateur
    return golden_songs


def popbutfy_statistics(data_dir):
    singers = []
    songs = []
    singer2songs = defaultdict(lambda: defaultdict(list))

    data_infos = glob(data_dir + "/*")

    for data_info in data_infos:
        data_info_split = data_info.split("/")[-1].split("#")

        singer, song = data_info_split[0], data_info_split[-1]
        singers.append(singer)
        songs.append(song)

        utts = glob(data_info + "/*")

        for utt in utts:
            uid = utt.split("/")[-1].split("_")[-1].split(".")[0]
            singer2songs[singer][song].append(uid)

    unique_singers = list(set(singers))
    unique_songs = list(set(songs))
    unique_singers.sort()
    unique_songs.sort()

    print(
        "PopBuTFy: {} singers, {} utterances ({} unique songs)".format(
            len(unique_singers), len(songs), len(unique_songs)
        )
    )
    print("Singers: \n{}".format("\t".join(unique_singers)))
    return singer2songs, unique_singers


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for popbutfy...\n")

    save_dir = os.path.join(output_path, "popbutfy")
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
    popbutfy_dir = dataset_path

    singer2songs, unique_singers = popbutfy_statistics(popbutfy_dir)
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
            for chosen_uid in songs[chosen_song]:
                res = {
                    "Dataset": "popbutfy",
                    "Singer": singer,
                    "Song": chosen_song,
                    "Uid": "{}#{}#".format(singer, chosen_song, chosen_uid),
                }
                res["Path"] = "{}#singing#{}/{}#singing#{}_{}.mp3".format(
                    singer, chosen_song, singer, chosen_song, chosen_uid
                )
                if not os.path.exists(os.path.join(popbutfy_dir, res["Path"])):
                    res["Path"] = "{}#singing#{}/{}#singing#{}_{}.wav".format(
                        singer, chosen_song, singer, chosen_song, chosen_uid
                    )
                res["Path"] = os.path.join(popbutfy_dir, res["Path"])
                assert os.path.exists(res["Path"])

                if res["Path"].split("/")[-1].split(".")[-1] == "wav":
                    waveform, sample_rate = torchaudio.load(res["Path"])
                    duration = waveform.size(-1) / sample_rate
                else:
                    waveform, sample_rate = librosa.load(res["Path"])
                    duration = waveform.shape[-1] / sample_rate
                res["Duration"] = duration

                if ([singer, chosen_song]) in test_songs:
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
