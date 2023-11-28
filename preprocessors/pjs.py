# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm
import glob
import json
import torchaudio

from utils.util import has_existed
from utils.io import save_audio


def get_splitted_utterances(
    raw_wav_dir, trimed_wav_dir, n_utterance_splits, overlapping
):
    res = []
    raw_song_files = glob.glob(
        os.path.join(raw_wav_dir, "**/pjs*_song.wav"), recursive=True
    )
    trimed_song_files = glob.glob(
        os.path.join(trimed_wav_dir, "**/*.wav"), recursive=True
    )

    if len(raw_song_files) * n_utterance_splits == len(trimed_song_files):
        print("Splitted done...")
        for wav_file in tqdm(trimed_song_files):
            uid = wav_file.split("/")[-1].split(".")[0]
            utt = {"Dataset": "pjs", "Singer": "male1", "Uid": uid, "Path": wav_file}

            waveform, sample_rate = torchaudio.load(wav_file)
            duration = waveform.size(-1) / sample_rate
            utt["Duration"] = duration

            res.append(utt)

    else:
        for wav_file in tqdm(raw_song_files):
            song_id = wav_file.split("/")[-1].split(".")[0]

            waveform, sample_rate = torchaudio.load(wav_file)
            trimed_waveform = torchaudio.functional.vad(waveform, sample_rate)
            trimed_waveform = torchaudio.functional.vad(
                trimed_waveform.flip(dims=[1]), sample_rate
            ).flip(dims=[1])

            audio_len = trimed_waveform.size(-1)
            lapping_len = overlapping * sample_rate

            for i in range(n_utterance_splits):
                start = i * audio_len // 3
                end = start + audio_len // 3 + lapping_len
                splitted_waveform = trimed_waveform[:, start:end]

                utt = {
                    "Dataset": "pjs",
                    "Singer": "male1",
                    "Uid": "{}_{}".format(song_id, i),
                }

                # Duration
                duration = splitted_waveform.size(-1) / sample_rate
                utt["Duration"] = duration

                # Save trimed wav
                splitted_waveform_file = os.path.join(
                    trimed_wav_dir, "{}.wav".format(utt["Uid"])
                )
                save_audio(splitted_waveform_file, splitted_waveform, sample_rate)

                # Path
                utt["Path"] = splitted_waveform_file

                res.append(utt)

    res = sorted(res, key=lambda x: x["Uid"])
    return res


def main(output_path, dataset_path, n_utterance_splits=3, overlapping=1):
    """
    1. Split one raw utterance to three splits (since some samples are too long)
    2. Overlapping of ajacent splits is 1 s
    """
    print("-" * 10)
    print("Preparing training dataset for PJS...")

    save_dir = os.path.join(output_path, "pjs")
    raw_wav_dir = os.path.join(dataset_path, "PJS_corpus_ver1.1")

    # Trim for silence
    trimed_wav_dir = os.path.join(dataset_path, "trim")
    os.makedirs(trimed_wav_dir, exist_ok=True)

    # Total utterances
    utterances = get_splitted_utterances(
        raw_wav_dir, trimed_wav_dir, n_utterance_splits, overlapping
    )
    total_uids = [utt["Uid"] for utt in utterances]

    # Test uids
    n_test_songs = 3
    test_uids = []
    for i in range(1, n_test_songs + 1):
        test_uids += [
            "pjs00{}_song_{}".format(i, split_id)
            for split_id in range(n_utterance_splits)
        ]

    # Train uids
    train_uids = [uid for uid in total_uids if uid not in test_uids]

    for dataset_type in ["train", "test"]:
        output_file = os.path.join(save_dir, "{}.json".format(dataset_type))
        if has_existed(output_file):
            continue

        uids = eval("{}_uids".format(dataset_type))
        res = [utt for utt in utterances if utt["Uid"] in uids]
        for i in range(len(res)):
            res[i]["index"] = i

        time = sum([utt["Duration"] for utt in res])
        print(
            "{}, Total size: {}, Total Duraions = {} s = {:.2f} hour\n".format(
                dataset_type, len(res), time, time / 3600
            )
        )

        # Save
        os.makedirs(save_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
