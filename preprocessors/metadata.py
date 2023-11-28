# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from tqdm import tqdm


def cal_metadata(cfg):
    """
    Dump metadata (singers.json, meta_info.json, utt2singer) for singer dataset or multi-datasets.
    """
    from collections import Counter

    datasets = cfg.dataset

    print("-" * 10)
    print("Preparing metadata...")
    print("Including: \n{}\n".format("\n".join(datasets)))

    datasets.sort()

    for dataset in tqdm(datasets):
        save_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        assert os.path.exists(save_dir)

        # 'train.json' and 'test.json' of target dataset
        train_metadata = os.path.join(save_dir, "train.json")
        test_metadata = os.path.join(save_dir, "test.json")

        # Sort the metadata as the duration order
        with open(train_metadata, "r", encoding="utf-8") as f:
            train_utterances = json.load(f)
        with open(test_metadata, "r", encoding="utf-8") as f:
            test_utterances = json.load(f)

        train_utterances = sorted(train_utterances, key=lambda x: x["Duration"])
        test_utterances = sorted(test_utterances, key=lambda x: x["Duration"])

        # Write back the sorted metadata
        with open(train_metadata, "w") as f:
            json.dump(train_utterances, f, indent=4, ensure_ascii=False)
        with open(test_metadata, "w") as f:
            json.dump(test_utterances, f, indent=4, ensure_ascii=False)

        # Paths of metadata needed to be generated
        singer_dict_file = os.path.join(save_dir, cfg.preprocess.spk2id)
        utt2singer_file = os.path.join(save_dir, cfg.preprocess.utt2spk)

        # Get the total duration and singer names for train and test utterances
        train_total_duration = sum(utt["Duration"] for utt in train_utterances)
        test_total_duration = sum(utt["Duration"] for utt in test_utterances)

        singer_names = set(
            f"{replace_augment_name(utt['Dataset'])}_{utt['Singer']}"
            for utt in train_utterances + test_utterances
        )

        # Write the utt2singer file and sort the singer names
        with open(utt2singer_file, "w", encoding="utf-8") as f:
            for utt in train_utterances + test_utterances:
                f.write(
                    f"{utt['Dataset']}_{utt['Uid']}\t{replace_augment_name(utt['Dataset'])}_{utt['Singer']}\n"
                )

        singer_names = sorted(singer_names)
        singer_lut = {name: i for i, name in enumerate(singer_names)}

        # dump singers.json
        with open(singer_dict_file, "w", encoding="utf-8") as f:
            json.dump(singer_lut, f, indent=4, ensure_ascii=False)

        meta_info = {
            "dataset": dataset,
            "statistics": {
                "size": len(train_utterances) + len(test_utterances),
                "hours": round(train_total_duration / 3600, 4)
                + round(test_total_duration / 3600, 4),
            },
            "train": {
                "size": len(train_utterances),
                "hours": round(train_total_duration / 3600, 4),
            },
            "test": {
                "size": len(test_utterances),
                "hours": round(test_total_duration / 3600, 4),
            },
            "singers": {"size": len(singer_lut)},
        }
        # Use Counter to count the minutes for each singer
        total_singer2mins = Counter()
        training_singer2mins = Counter()
        for utt in train_utterances:
            k = f"{replace_augment_name(utt['Dataset'])}_{utt['Singer']}"
            training_singer2mins[k] += utt["Duration"] / 60
            total_singer2mins[k] += utt["Duration"] / 60
        for utt in test_utterances:
            k = f"{replace_augment_name(utt['Dataset'])}_{utt['Singer']}"
            total_singer2mins[k] += utt["Duration"] / 60

        training_singer2mins = dict(
            sorted(training_singer2mins.items(), key=lambda x: x[1], reverse=True)
        )
        training_singer2mins = {k: round(v, 2) for k, v in training_singer2mins.items()}
        meta_info["singers"]["training_minutes"] = training_singer2mins

        total_singer2mins = dict(
            sorted(total_singer2mins.items(), key=lambda x: x[1], reverse=True)
        )
        total_singer2mins = {k: round(v, 2) for k, v in total_singer2mins.items()}
        meta_info["singers"]["minutes"] = total_singer2mins

        with open(os.path.join(save_dir, "meta_info.json"), "w") as f:
            json.dump(meta_info, f, indent=4, ensure_ascii=False)

        for singer, min in training_singer2mins.items():
            print(f"Singer {singer}: {min} mins for training")
        print("-" * 10, "\n")


def replace_augment_name(dataset: str) -> str:
    """Replace the augmented dataset name with the original dataset name.
    >>> print(replace_augment_name("dataset_equalizer"))
    dataset
    """
    if "equalizer" in dataset:
        dataset = dataset.replace("_equalizer", "")
    elif "formant_shift" in dataset:
        dataset = dataset.replace("_formant_shift", "")
    elif "pitch_shift" in dataset:
        dataset = dataset.replace("_pitch_shift", "")
    elif "time_stretch" in dataset:
        dataset = dataset.replace("_time_stretch", "")
    else:
        pass
    return dataset
