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


from utils.io import save_audio
from utils.util import has_existed
from utils.audio_slicer import Slicer
from preprocessors import GOLDEN_TEST_SAMPLES


def split_to_utterances(dataset_path, singer, style, output_dir):
    data_dir = os.path.join(dataset_path, singer, style)

    print("Splitting to utterances for {}...".format(data_dir))

    wave_files = glob(data_dir + "/*.wav")

    for wav_file in tqdm(wave_files):
        # Load waveform
        song_name = wav_file.split("/")[-1].split(".")[0]
        waveform, fs = torchaudio.load(wav_file)

        # Split
        slicer = Slicer(sr=fs, threshold=-40.0, max_sil_kept=4000)
        chunks = slicer.slice(waveform)

        for i, chunk in enumerate(chunks):
            save_dir = os.path.join(output_dir, singer, style, song_name)
            os.makedirs(save_dir, exist_ok=True)

            output_file = os.path.join(save_dir, "{:04d}.wav".format(i))
            save_audio(output_file, chunk, fs)


def _main(dataset_path):
    """
    Split to utterances
    """
    utterance_dir = os.path.join(dataset_path, "utterances")

    singer_infos = glob(dataset_path + "/*")

    for singer_info in singer_infos:
        singer = singer_info.split("/")[-1]

        for style in ["read", "sing"]:
            split_to_utterances(dataset_path, singer, style, utterance_dir)


def get_test_songs():
    golden_samples = GOLDEN_TEST_SAMPLES["nus48e"]
    # every item is a tuple (singer, song)
    golden_songs = [s.split("#")[:2] for s in golden_samples]
    # singer_song, eg: Female1#Almost_lover_Amateur
    return golden_songs


def nus48e_statistics(data_dir):
    singers = []
    songs = []
    singer2songs = defaultdict(lambda: defaultdict(list))

    singer_infos = glob(data_dir + "/*")

    for singer_info in singer_infos:
        singer_info_split = singer_info.split("/")[-1]

        style_infos = glob(singer_info + "/*")

        for style_info in style_infos:
            style_info_split = style_info.split("/")[-1]

            singer = singer_info_split + "_" + style_info_split
            singers.append(singer)

            song_infos = glob(style_info + "/*")

            for song_info in song_infos:
                song = song_info.split("/")[-1]

                songs.append(song)

                utts = glob(song_info + "/*.wav")

                for utt in utts:
                    uid = utt.split("/")[-1].split(".")[0]
                    singer2songs[singer][song].append(uid)

    unique_singers = list(set(singers))
    unique_songs = list(set(songs))
    unique_singers.sort()
    unique_songs.sort()

    print(
        "nus_48_e: {} singers, {} utterances ({} unique songs)".format(
            len(unique_singers), len(songs), len(unique_songs)
        )
    )
    print("Singers: \n{}".format("\t".join(unique_singers)))
    return singer2songs, unique_singers


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for nus48e...\n")

    if not os.path.exists(os.path.join(dataset_path, "utterances")):
        print("Spliting into utterances...\n")
        _main(dataset_path)

    save_dir = os.path.join(output_path, "nus48e")
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
    nus48e_path = os.path.join(dataset_path, "utterances")

    singer2songs, unique_singers = nus48e_statistics(nus48e_path)
    test_songs = get_test_songs()

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for singer, songs in singer2songs.items():
        song_names = list(songs.keys())

        for chosen_song in song_names:
            for chosen_uid in songs[chosen_song]:
                res = {
                    "Dataset": "nus48e",
                    "Singer": singer,
                    "Uid": "{}#{}#{}".format(singer, chosen_song, chosen_uid),
                }
                res["Path"] = "{}/{}/{}/{}.wav".format(
                    singer.split("_")[0], singer.split("_")[-1], chosen_song, chosen_uid
                )
                res["Path"] = os.path.join(nus48e_path, res["Path"])
                assert os.path.exists(res["Path"])

                waveform, sample_rate = torchaudio.load(res["Path"])
                duration = waveform.size(-1) / sample_rate
                res["Duration"] = duration

                if duration <= 1e-8:
                    continue

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
