# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import json
import torchaudio
from tqdm import tqdm
from collections import defaultdict


from utils.io import save_audio
from utils.util import has_existed, remove_and_create
from utils.audio_slicer import Slicer
from preprocessors import GOLDEN_TEST_SAMPLES


def split_to_utterances(input_dir, output_dir):
    print("Splitting to utterances for {}...".format(input_dir))

    files_list = glob.glob("*.flac", root_dir=input_dir)
    files_list.sort()
    for wav_file in tqdm(files_list):
        # Load waveform
        waveform, fs = torchaudio.load(os.path.join(input_dir, wav_file))

        # Song name
        filename = wav_file.replace(" ", "")
        filename = filename.replace("(Live)", "")
        song_id, filename = filename.split("李健-")

        song_id = song_id.split("_")[0]
        song_name = "{:03d}".format(int(song_id)) + filename.split("_")[0].split("-")[0]

        # Split
        slicer = Slicer(sr=fs, threshold=-30.0, max_sil_kept=3000)
        chunks = slicer.slice(waveform)

        save_dir = os.path.join(output_dir, song_name)
        remove_and_create(save_dir)

        for i, chunk in enumerate(chunks):
            output_file = os.path.join(save_dir, "{:04d}.wav".format(i))
            save_audio(output_file, chunk, fs)


def _main(dataset_path):
    """
    Split to utterances
    """
    utterance_dir = os.path.join(dataset_path, "utterances")
    split_to_utterances(os.path.join(dataset_path, "vocal_v2"), utterance_dir)


def get_test_songs():
    golden_samples = GOLDEN_TEST_SAMPLES["lijian"]
    golden_songs = [s.split("_")[0] for s in golden_samples]
    return golden_songs


def statistics(utt_dir):
    song2utts = defaultdict(list)

    song_infos = glob.glob(utt_dir + "/*")
    song_infos.sort()
    for song in song_infos:
        song_name = song.split("/")[-1]
        utt_infos = glob.glob(song + "/*.wav")
        utt_infos.sort()
        for utt in utt_infos:
            uid = utt.split("/")[-1].split(".")[0]
            song2utts[song_name].append(uid)

    utt_sum = sum([len(utts) for utts in song2utts.values()])
    print("Li Jian: {} unique songs, {} utterances".format(len(song2utts), utt_sum))
    return song2utts


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for Li Jian...\n")

    if not os.path.exists(os.path.join(dataset_path, "utterances")):
        print("Spliting into utterances...\n")
        _main(dataset_path)

    save_dir = os.path.join(output_path, "lijian")
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    if has_existed(test_output_file):
        return

    # Load
    lijian_path = os.path.join(dataset_path, "utterances")
    song2utts = statistics(lijian_path)
    test_songs = get_test_songs()

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for chosen_song, utts in tqdm(song2utts.items()):
        for chosen_uid in song2utts[chosen_song]:
            res = {
                "Dataset": "lijian",
                "Singer": "lijian",
                "Uid": "{}_{}".format(chosen_song, chosen_uid),
            }
            res["Path"] = "{}/{}.wav".format(chosen_song, chosen_uid)
            res["Path"] = os.path.join(lijian_path, res["Path"])
            assert os.path.exists(res["Path"])

            waveform, sample_rate = torchaudio.load(res["Path"])
            duration = waveform.size(-1) / sample_rate
            res["Duration"] = duration

            if duration <= 1e-8:
                continue

            if chosen_song in test_songs:
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
