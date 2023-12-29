# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
import os
import json
import torchaudio
from tqdm import tqdm
from collections import defaultdict

from utils.util import has_existed


def statistics(utterance_dir):
    singers = []
    songs = []
    utts_all = []
    singers2songs = defaultdict(lambda: defaultdict(list))

    singer_infos = glob(utterance_dir + "/*")

    for singer_info in singer_infos:
        singer = singer_info.split("/")[-1]

        song_infos = glob(singer_info + "/*")

        for song_info in song_infos:
            song = song_info.split("/")[-1]

            singers.append(singer)
            songs.append(song)

            utts = glob(song_info + "/*.wav")
            utts_all.extend(utts)

            for utt in utts:
                uid = utt.split("/")[-1].split(".")[0]
                singers2songs[singer][song].append(uid)

    unique_singers = list(set(singers))
    unique_songs = list(set(songs))
    unique_singers.sort()
    unique_songs.sort()

    print(
        "Statistics: {} singers, {} utterances ({} unique songs)".format(
            len(unique_singers), len(utts_all), len(unique_songs)
        )
    )
    print("Singers: \n{}".format("\t".join(unique_singers)))
    return singers2songs, unique_singers


def main(output_path, dataset_path, dataset_name):
    print("-" * 10)
    print("Preparing samples for {}...\n".format(dataset_name))

    save_dir = os.path.join(output_path, dataset_name)
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
    singers2songs, unique_singers = statistics(dataset_path)

    # We select songs of standard samples as test songs
    train = []
    test = []
    test_songs = set()

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for singer, songs in singers2songs.items():
        song_names = list(songs.keys())

        print("Singer {}...".format(singer))
        for chosen_song in tqdm(song_names):
            for chosen_uid in songs[chosen_song]:
                res = {
                    "Dataset": dataset_name,
                    "Singer": singer,
                    "Uid": "{}_{}_{}".format(singer, chosen_song, chosen_uid),
                }
                res["Path"] = "{}/{}/{}.wav".format(singer, chosen_song, chosen_uid)
                res["Path"] = os.path.join(dataset_path, res["Path"])
                assert os.path.exists(res["Path"])

                waveform, sample_rate = torchaudio.load(res["Path"])
                duration = waveform.size(-1) / sample_rate
                res["Duration"] = duration

                # Remove the utterance whose duration is shorter than 0.1s
                if duration <= 1e-2:
                    continue

                # Place into train or test
                if "{}_{}".format(singer, chosen_song) not in test_songs:
                    test_songs.add("{}_{}".format(singer, chosen_song))

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
