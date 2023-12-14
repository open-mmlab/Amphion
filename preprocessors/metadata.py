# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from tqdm import tqdm


def cal_metadata(cfg, dataset_types=["train", "test"]):
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

        # 'train.json' and 'test.json' and 'valid.json' of target dataset
        meta_info = dict()
        utterances_dict = dict()
        all_utterances = list()
        duration = dict()
        total_duration = 0.0
        for dataset_type in dataset_types:
            metadata = os.path.join(save_dir, "{}.json".format(dataset_type))

            # Sort the metadata as the duration order
            with open(metadata, "r", encoding="utf-8") as f:
                utterances = json.load(f)
            utterances = sorted(utterances, key=lambda x: x["Duration"])
            utterances_dict[dataset_type] = utterances
            all_utterances.extend(utterances)

            # Write back the sorted metadata
            with open(metadata, "w") as f:
                json.dump(utterances, f, indent=4, ensure_ascii=False)

            # Get the total duration and singer names for train and test utterances
            duration[dataset_type] = sum(utt["Duration"] for utt in utterances)
            total_duration += duration[dataset_type]

        # Paths of metadata needed to be generated
        singer_dict_file = os.path.join(save_dir, cfg.preprocess.spk2id)
        utt2singer_file = os.path.join(save_dir, cfg.preprocess.utt2spk)

        singer_names = set(
            f"{replace_augment_name(utt['Dataset'])}_{utt['Singer']}"
            for utt in all_utterances
        )

        # Write the utt2singer file and sort the singer names
        with open(utt2singer_file, "w", encoding="utf-8") as f:
            for utt in all_utterances:
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
                "size": len(all_utterances),
                "hours": round(total_duration / 3600, 4),
            },
        }

        for dataset_type in dataset_types:
            meta_info[dataset_type] = {
                "size": len(utterances_dict[dataset_type]),
                "hours": round(duration[dataset_type] / 3600, 4),
            }

        meta_info["singers"] = {"size": len(singer_lut)}

        # Use Counter to count the minutes for each singer
        total_singer2mins = Counter()
        training_singer2mins = Counter()
        for dataset_type in dataset_types:
            for utt in utterances_dict[dataset_type]:
                k = f"{replace_augment_name(utt['Dataset'])}_{utt['Singer']}"
                if dataset_type == "train":
                    training_singer2mins[k] += utt["Duration"] / 60
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
            print(f"Speaker/Singer {singer}: {min} mins for training")
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
