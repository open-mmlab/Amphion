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

def hifitts_statistics(data_dir):
    speakers = []
    distribution2books2utts = defaultdict(
        lambda:defaultdict(list)
    )
    distribution_infos = glob(data_dir + "/*.json")
    
    for distribution_info in distribution_infos:
        distribution = distribution_info.split("/")[-1].split(".")[0]
        speaker_id = distribution.split("_")[0] 
        speakers.append(speaker_id)

        with open(distribution_info, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                text_normalized = entry.get("text_normalized")
                audio_path = entry.get("audio_filepath")
                book = audio_path.split("/")[-2]
                distribution2books2utts[distribution][book].append((text_normalized, audio_path))

    unique_speakers = list(set(speakers))
    unique_speakers.sort()

    print("Speakers: \n{}".format("\t".join(unique_speakers)))
    return distribution2books2utts, unique_speakers


def main(output_path, dataset_path):
    print("-" * 10)
    print("Preparing samples for hifitts...\n")

    save_dir = os.path.join(output_path, "hifitts")
    os.makedirs(save_dir, exist_ok=True)
    print('Saving to ', save_dir)
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    valid_output_file = os.path.join(save_dir, "valid.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    if has_existed(train_output_file):
        return
    utt2singer = open(utt2singer_file, "w")

    # Load
    hifitts_path = dataset_path

    distribution2books2utts, unique_speakers = hifitts_statistics(
        hifitts_path
    )
    
    train = []
    test = []
    valid = []

    train_index_count = 0
    test_index_count = 0
    valid_index_count = 0

    train_total_duration = 0
    test_total_duration = 0
    valid_total_duration = 0
    
    for distribution, books2utts in tqdm(
        distribution2books2utts.items(),
        desc=f"Distribution"
    ):
        speaker = distribution.split("_")[0]
        book_names = list(books2utts.keys())
        for chosen_book in tqdm(book_names, desc=f"chosen_book"):
            for text, utt_path in tqdm(books2utts[chosen_book], desc=f"utterance"):
                chosen_uid = utt_path.split("/")[-1].split(".")[0]
                res = {
                    "Dataset":"hifitts",
                    "Singer":speaker,
                    "Uid": "{}#{}#{}#{}".format(
                            distribution, speaker, chosen_book, chosen_uid
                        ),
                    "Text": text
                }

                res["Path"] = os.path.join(hifitts_path, utt_path)
                assert os.path.exists(res["Path"])

                waveform, sample_rate = torchaudio.load(res["Path"])
                duration = waveform.size(-1) / sample_rate
                res["Duration"] = duration

                if "train" in distribution:
                    res["index"] = train_index_count
                    train_total_duration += duration
                    train.append(res)
                    train_index_count += 1

                elif 'test' in distribution:
                    res["index"] = test_index_count
                    test_total_duration += duration
                    test.append(res)
                    test_index_count += 1
                    
                elif 'dev' in distribution:
                    res["index"] = valid_index_count
                    valid_total_duration += duration
                    valid.append(res)
                    valid_index_count += 1

                utt2singer.write("{}\t{}\n".format(res["Uid"], res["Singer"]))

    print("#Train = {}, #Test = {}, #Valid = {}".format(len(train), len(test), len(valid)))
    print(
        "#Train hours= {}, #Test hours= {}, #Valid hours= {}".format(
            train_total_duration / 3600, test_total_duration / 3600, valid_total_duration / 3600
        )
    )

    # Save train.json, test.json, valid.json
    with open(train_output_file, "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(test_output_file, "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
    with open(valid_output_file, "w") as f:
        json.dump(valid, f, indent=4, ensure_ascii=False)

    # Save singers.json
    singer_lut = {name: i for i, name in enumerate(unique_speakers)}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)