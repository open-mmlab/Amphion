# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import os
from collections import defaultdict
from tqdm import tqdm


def get_uids_and_wav_paths(cfg, dataset, dataset_type):
    assert dataset == "bigdata"
    dataset_dir = os.path.join(
        cfg.OUTPUT_PATH,
        "preprocess/{}_version".format(cfg.PREPROCESS_VERSION),
        "bigdata/{}".format(cfg.BIGDATA_VERSION),
    )
    dataset_file = os.path.join(
        dataset_dir, "{}.json".format(dataset_type.split("_")[-1])
    )
    with open(dataset_file, "r") as f:
        utterances = json.load(f)

    # Uids
    uids = [u["Uid"] for u in utterances]

    # Wav paths
    wav_paths = [u["Path"] for u in utterances]

    return uids, wav_paths


def take_duration(utt):
    return utt["Duration"]


def main(output_path, cfg):
    datasets = cfg.dataset

    print("-" * 10)
    print("Preparing samples for bigdata...")
    print("Including: \n{}\n".format("\n".join(datasets)))

    datasets.sort()
    bigdata_version = "_".join(datasets)

    save_dir = os.path.join(output_path, bigdata_version)
    os.makedirs(save_dir, exist_ok=True)

    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    singer_dict_file = os.path.join(save_dir, cfg.preprocess.spk2id)
    utt2singer_file = os.path.join(save_dir, cfg.preprocess.utt2spk)
    utt2singer = open(utt2singer_file, "a+")
    # We select songs of standard samples as test songs
    train = []
    test = []

    train_total_duration = 0
    test_total_duration = 0

    # Singer unique names
    singer_names = set()

    for dataset in datasets:
        dataset_path = os.path.join(output_path, dataset)
        train_json = os.path.join(dataset_path, "train.json")
        test_json = os.path.join(dataset_path, "test.json")

        with open(train_json, "r", encoding="utf-8") as f:
            train_utterances = json.load(f)

        with open(test_json, "r", encoding="utf-8") as f:
            test_utterances = json.load(f)

        for utt in tqdm(train_utterances):
            train.append(utt)
            train_total_duration += utt["Duration"]
            singer_names.add("{}_{}".format(utt["Dataset"], utt["Singer"]))
            utt2singer.write(
                "{}_{}\t{}_{}\n".format(
                    utt["Dataset"], utt["Uid"], utt["Dataset"], utt["Singer"]
                )
            )

        for utt in test_utterances:
            test.append(utt)
            test_total_duration += utt["Duration"]
            singer_names.add("{}_{}".format(utt["Dataset"], utt["Singer"]))
            utt2singer.write(
                "{}_{}\t{}_{}\n".format(
                    utt["Dataset"], utt["Uid"], utt["Dataset"], utt["Singer"]
                )
            )

    utt2singer.close()

    train.sort(key=take_duration)
    test.sort(key=take_duration)
    print("#Train = {}, #Test = {}".format(len(train), len(test)))
    print(
        "#Train hours= {}, #Test hours= {}".format(
            train_total_duration / 3600, test_total_duration / 3600
        )
    )

    # Singer Look Up Table
    singer_names = list(singer_names)
    singer_names.sort()
    singer_lut = {name: i for i, name in enumerate(singer_names)}
    print("#Singers: {}\n".format(len(singer_lut)))

    # Save
    with open(train_output_file, "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(test_output_file, "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)

    # Save meta info
    meta_info = {
        "datasets": datasets,
        "train": {"size": len(train), "hours": round(train_total_duration / 3600, 4)},
        "test": {"size": len(test), "hours": round(test_total_duration / 3600, 4)},
        "singers": {"size": len(singer_lut)},
    }
    singer2mins = defaultdict(float)
    for utt in train:
        dataset, singer, duration = utt["Dataset"], utt["Singer"], utt["Duration"]
        singer2mins["{}_{}".format(dataset, singer)] += duration / 60
    singer2mins = sorted(singer2mins.items(), key=lambda x: x[1], reverse=True)
    singer2mins = dict(
        zip([i[0] for i in singer2mins], [round(i[1], 2) for i in singer2mins])
    )
    meta_info["singers"]["training_minutes"] = singer2mins

    with open(os.path.join(save_dir, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=4, ensure_ascii=False)

    for singer, min in singer2mins.items():
        print("Singer {}: {} mins".format(singer, min))
    print("-" * 10, "\n")
