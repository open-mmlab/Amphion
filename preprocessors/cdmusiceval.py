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

from utils.util import has_existed, remove_and_create
from utils.audio_slicer import split_utterances_from_audio


def split_to_utterances(input_dir, output_dir):
    print("Splitting to utterances for {}...".format(input_dir))

    files_list = glob("*", root_dir=input_dir)
    files_list.sort()
    for wav_file in tqdm(files_list):
        # # Load waveform
        # waveform, fs = torchaudio.load(os.path.join(input_dir, wav_file))

        # Singer name, Song name
        song_name, singer_name = wav_file.split("_")[2].split("-")
        save_dir = os.path.join(output_dir, singer_name, song_name)

        split_utterances_from_audio(
            os.path.join(input_dir, wav_file), save_dir, max_duration_of_utterance=10
        )

        # # Split
        # slicer = Slicer(sr=fs, threshold=-30.0, max_sil_kept=3000, min_interval=1000)
        # chunks = slicer.slice(waveform)

        # for i, chunk in enumerate(chunks):
        #     save_dir = os.path.join(output_dir, singer_name, song_name)
        #     os.makedirs(save_dir, exist_ok=True)

        #     output_file = os.path.join(save_dir, "{:04d}.wav".format(i))
        #     save_audio(output_file, chunk, fs)


def _main(dataset_path):
    """
    Split to utterances
    """
    utterance_dir = os.path.join(dataset_path, "utterances")
    remove_and_create(utterance_dir)
    split_to_utterances(os.path.join(dataset_path, "vocal"), utterance_dir)


def statistics(utterance_dir):
    singers = []
    songs = []
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

            for utt in utts:
                uid = utt.split("/")[-1].split(".")[0]
                singers2songs[singer][song].append(uid)

    unique_singers = list(set(singers))
    unique_songs = list(set(songs))
    unique_singers.sort()
    unique_songs.sort()

    print(
        "Statistics: {} singers, {} utterances ({} unique songs)".format(
            len(unique_singers), len(songs), len(unique_songs)
        )
    )
    print("Singers: \n{}".format("\t".join(unique_singers)))
    return singers2songs, unique_singers


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing samples for CD Music Eval...\n")

    if not os.path.exists(os.path.join(dataset_path, "utterances")):
        print("Spliting into utterances...\n")
        _main(dataset_path)

    save_dir = os.path.join(output_path, "cdmusiceval")
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
    utt_path = os.path.join(dataset_path, "utterances")
    singers2songs, unique_singers = statistics(utt_path)

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for singer, songs in tqdm(singers2songs.items()):
        song_names = list(songs.keys())

        for chosen_song in song_names:
            for chosen_uid in songs[chosen_song]:
                res = {
                    "Dataset": "cdmusiceval",
                    "Singer": singer,
                    "Uid": "{}_{}_{}".format(singer, chosen_song, chosen_uid),
                }
                res["Path"] = "{}/{}/{}.wav".format(singer, chosen_song, chosen_uid)
                res["Path"] = os.path.join(utt_path, res["Path"])
                assert os.path.exists(res["Path"])

                waveform, sample_rate = torchaudio.load(res["Path"])
                duration = waveform.size(-1) / sample_rate
                res["Duration"] = duration

                if duration <= 1e-8:
                    continue

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
