# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import os
import glob
from tqdm import tqdm
import torchaudio
import pandas as pd
from glob import glob
from collections import defaultdict

from utils.io import save_audio
from utils.util import has_existed
from preprocessors import GOLDEN_TEST_SAMPLES


def save_utterance(output_file, waveform, fs, start, end, overlap=0.1):
    """
    waveform: [#channel, audio_len]
    start, end, overlap: seconds
    """
    start = int((start - overlap) * fs)
    end = int((end + overlap) * fs)
    utterance = waveform[:, start:end]
    save_audio(output_file, utterance, fs)


def split_to_utterances(language_dir, output_dir):
    print("Splitting to utterances for {}...".format(language_dir))
    wav_dir = os.path.join(language_dir, "wav")
    phoneme_dir = os.path.join(language_dir, "txt")
    annot_dir = os.path.join(language_dir, "csv")

    pitches = set()
    for wav_file in tqdm(glob("{}/*.wav".format(wav_dir))):
        # Load waveform
        song_name = wav_file.split("/")[-1].split(".")[0]
        waveform, fs = torchaudio.load(wav_file)

        # Load utterances
        phoneme_file = os.path.join(phoneme_dir, "{}.txt".format(song_name))
        with open(phoneme_file, "r") as f:
            lines = f.readlines()
        utterances = [l.strip().split() for l in lines]
        utterances = [utt for utt in utterances if len(utt) > 0]

        # Load annotation
        annot_file = os.path.join(annot_dir, "{}.csv".format(song_name))
        annot_df = pd.read_csv(annot_file)
        pitches = pitches.union(set(annot_df["pitch"]))
        starts = annot_df["start"].tolist()
        ends = annot_df["end"].tolist()
        syllables = annot_df["syllable"].tolist()

        # Split
        curr = 0
        for i, phones in enumerate(utterances):
            sz = len(phones)
            assert phones[0] == syllables[curr]
            assert phones[-1] == syllables[curr + sz - 1]

            s = starts[curr]
            e = ends[curr + sz - 1]
            curr += sz

            save_dir = os.path.join(output_dir, song_name)
            os.makedirs(save_dir, exist_ok=True)

            output_file = os.path.join(save_dir, "{:04d}.wav".format(i))
            save_utterance(output_file, waveform, fs, start=s, end=e)


def _main(dataset_path):
    """
    Split to utterances
    """
    utterance_dir = os.path.join(dataset_path, "utterances")

    for lang in ["english", "korean"]:
        split_to_utterances(os.path.join(dataset_path, lang), utterance_dir)


def get_test_songs():
    golden_samples = GOLDEN_TEST_SAMPLES["csd"]
    # every item is a tuple (language, song)
    golden_songs = [s.split("_")[:2] for s in golden_samples]
    # language_song, eg: en_001a
    return golden_songs


def csd_statistics(data_dir):
    languages = []
    songs = []
    languages2songs = defaultdict(lambda: defaultdict(list))

    folder_infos = glob(data_dir + "/*")

    for folder_info in folder_infos:
        folder_info_split = folder_info.split("/")[-1]

        language = folder_info_split[:2]
        song = folder_info_split[2:]

        languages.append(language)
        songs.append(song)

        utts = glob(folder_info + "/*")

        for utt in utts:
            uid = utt.split("/")[-1].split(".")[0]
            languages2songs[language][song].append(uid)

    unique_languages = list(set(languages))
    unique_songs = list(set(songs))
    unique_languages.sort()
    unique_songs.sort()

    print(
        "csd: {} languages, {} utterances ({} unique songs)".format(
            len(unique_languages), len(songs), len(unique_songs)
        )
    )
    print("Languages: \n{}".format("\t".join(unique_languages)))
    return languages2songs


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing test samples for csd...\n")

    if not os.path.exists(os.path.join(dataset_path, "utterances")):
        print("Spliting into utterances...\n")
        _main(dataset_path)

    save_dir = os.path.join(output_path, "csd")
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    if has_existed(test_output_file):
        return

    # Load
    csd_path = os.path.join(dataset_path, "utterances")

    language2songs = csd_statistics(csd_path)
    test_songs = get_test_songs()

    # We select songs of standard samples as test songs
    train = []
    test = []

    train_index_count = 0
    test_index_count = 0

    train_total_duration = 0
    test_total_duration = 0

    for language, songs in tqdm(language2songs.items()):
        song_names = list(songs.keys())

        for chosen_song in song_names:
            for chosen_uid in songs[chosen_song]:
                res = {
                    "Dataset": "csd",
                    "Singer": "Female1_{}".format(language),
                    "Uid": "{}_{}_{}".format(language, chosen_song, chosen_uid),
                }
                res["Path"] = "{}{}/{}.wav".format(language, chosen_song, chosen_uid)
                res["Path"] = os.path.join(csd_path, res["Path"])
                assert os.path.exists(res["Path"])

                waveform, sample_rate = torchaudio.load(res["Path"])
                duration = waveform.size(-1) / sample_rate
                res["Duration"] = duration

                if [language, chosen_song] in test_songs:
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
