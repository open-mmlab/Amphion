# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import os
import json
import torchaudio
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from utils.util import has_existed
from utils.audio_slicer import split_utterances_from_audio
from preprocessors import GOLDEN_TEST_SAMPLES


def _split_utts():
    raw_dir = "/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xueyaozhang/dataset/李玟/cocoeval/raw"
    output_root = "/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xueyaozhang/dataset/李玟/cocoeval/utterances"

    if os.path.exists(output_root):
        os.system("rm -rf {}".format(output_root))

    vocal_files = glob(os.path.join(raw_dir, "*/vocal.wav"))
    for vocal_f in tqdm(vocal_files):
        song_name = vocal_f.split("/")[-2]

        output_dir = os.path.join(output_root, song_name)
        os.makedirs(output_dir, exist_ok=True)

        split_utterances_from_audio(vocal_f, output_dir, min_interval=300)


def cocoeval_statistics(data_dir):
    song2utts = defaultdict(list)

    song_infos = glob(data_dir + "/*")

    for song in song_infos:
        song_name = song.split("/")[-1]
        utts = glob(song + "/*.wav")
        for utt in utts:
            uid = utt.split("/")[-1].split(".")[0]
            song2utts[song_name].append(uid)

    print("Cocoeval: {} songs".format(len(song_infos)))
    return song2utts


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing datasets for Cocoeval...\n")

    save_dir = os.path.join(output_path, "cocoeval")
    test_output_file = os.path.join(save_dir, "test.json")
    if has_existed(test_output_file):
        return

    # Load
    song2utts = cocoeval_statistics(dataset_path)

    train, test = [], []
    train_index_count, test_index_count = 0, 0
    train_total_duration, test_total_duration = 0.0, 0.0

    for song_name, uids in tqdm(song2utts.items()):
        for chosen_uid in uids:
            res = {
                "Dataset": "cocoeval",
                "Singer": "TBD",
                "Song": song_name,
                "Uid": "{}_{}".format(song_name, chosen_uid),
            }
            res["Path"] = "{}/{}.wav".format(song_name, chosen_uid)
            res["Path"] = os.path.join(dataset_path, res["Path"])
            assert os.path.exists(res["Path"])

            waveform, sample_rate = torchaudio.load(res["Path"])
            duration = waveform.size(-1) / sample_rate
            res["Duration"] = duration

            res["index"] = test_index_count
            test_total_duration += duration
            test.append(res)
            test_index_count += 1

    print("#Train = {}, #Test = {}".format(len(train), len(test)))
    print(
        "#Train hours= {}, #Test hours= {}".format(
            train_total_duration / 3600, test_total_duration / 3600
        )
    )

    # Save
    os.makedirs(save_dir, exist_ok=True)
    with open(test_output_file, "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
