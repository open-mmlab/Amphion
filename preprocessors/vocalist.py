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


def vocalist_statistics(data_dir):
    singers = []
    songs = []
    global2singer2songs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    global_infos = glob(data_dir + "/*")

    for global_info in global_infos:
        global_split = global_info.split("/")[-1]

        singer_infos = glob(global_info + "/*")

        for singer_info in singer_infos:
            singer = singer_info.split("/")[-1]

            singers.append(singer)

            song_infos = glob(singer_info + "/*")
            for song_info in song_infos:
                song = song_info.split("/")[-1]

                songs.append(song)

                utts = glob(song_info + "/*.wav")

                for utt in utts:
                    uid = utt.split("/")[-1].split(".")[0]
                    global2singer2songs[global_split][singer][song].append(uid)

    unique_singers = list(set(singers))
    unique_songs = list(set(songs))
    unique_singers.sort()
    unique_songs.sort()

    print(
        "vocalist: {} singers, {} songs ({} unique songs)".format(
            len(unique_singers), len(songs), len(unique_songs)
        )
    )
    print("Singers: \n{}".format("\t".join(unique_singers)))
    return global2singer2songs, unique_singers


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for vocalist...\n")

    save_dir = os.path.join(output_path, "vocalist")
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
    vocalist_path = dataset_path

    global2singer2songs, unique_singers = vocalist_statistics(vocalist_path)

    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for global_info, singer2songs in tqdm(global2singer2songs.items()):
        for singer, songs in tqdm(singer2songs.items()):
            song_names = list(songs.keys())

            for chosen_song in song_names:
                for chosen_uid in songs[chosen_song]:
                    res = {
                        "Dataset": "opensinger",
                        "Singer": singer,
                        "Song": chosen_song,
                        "Uid": "{}_{}_{}".format(singer, chosen_song, chosen_uid),
                    }
                    res["Path"] = "{}/{}/{}/{}.wav".format(
                        global_info, singer, chosen_song, chosen_uid
                    )
                    res["Path"] = os.path.join(vocalist_path, res["Path"])
                    assert os.path.exists(res["Path"])

                    waveform, sample_rate = torchaudio.load(res["Path"])
                    duration = waveform.size(-1) / sample_rate
                    res["Duration"] = duration

                    res["index"] = test_index_count
                    test_total_duration += duration
                    test.append(res)
                    test_index_count += 1

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
