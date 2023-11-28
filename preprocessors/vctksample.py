# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pickle
import glob
from collections import defaultdict
from tqdm import tqdm
from preprocessors import get_golden_samples_indexes


TRAIN_MAX_NUM_EVERY_PERSON = 250
TEST_MAX_NUM_EVERY_PERSON = 25


def select_sample_idxs():
    # =========== Train ===========
    with open(os.path.join(vctk_dir, "train.json"), "r") as f:
        raw_train = json.load(f)

    train_idxs = []
    train_nums = defaultdict(int)
    for utt in tqdm(raw_train):
        idx = utt["index"]
        singer = utt["Singer"]

        if train_nums[singer] < TRAIN_MAX_NUM_EVERY_PERSON:
            train_idxs.append(idx)
            train_nums[singer] += 1

    # =========== Test ===========
    with open(os.path.join(vctk_dir, "test.json"), "r") as f:
        raw_test = json.load(f)

    # golden test
    test_idxs = get_golden_samples_indexes(
        dataset_name="vctk", split="test", dataset_dir=vctk_dir
    )
    test_nums = defaultdict(int)
    for idx in test_idxs:
        singer = raw_test[idx]["Singer"]
        test_nums[singer] += 1

    for utt in tqdm(raw_test):
        idx = utt["index"]
        singer = utt["Singer"]

        if test_nums[singer] < TEST_MAX_NUM_EVERY_PERSON:
            test_idxs.append(idx)
            test_nums[singer] += 1

    train_idxs.sort()
    test_idxs.sort()
    return train_idxs, test_idxs, raw_train, raw_test


if __name__ == "__main__":
    root_path = ""
    vctk_dir = os.path.join(root_path, "vctk")
    sample_dir = os.path.join(root_path, "vctksample")
    os.makedirs(sample_dir, exist_ok=True)

    train_idxs, test_idxs, raw_train, raw_test = select_sample_idxs()
    print("#Train = {}, #Test = {}".format(len(train_idxs), len(test_idxs)))

    for split, chosen_idxs, utterances in zip(
        ["train", "test"], [train_idxs, test_idxs], [raw_train, raw_test]
    ):
        print(
            "#{} = {}, #chosen idx = {}\n".format(
                split, len(utterances), len(chosen_idxs)
            )
        )

        # Select features
        feat_files = glob.glob(
            "**/{}.pkl".format(split), root_dir=vctk_dir, recursive=True
        )
        for file in tqdm(feat_files):
            raw_file = os.path.join(vctk_dir, file)
            new_file = os.path.join(sample_dir, file)

            new_dir = "/".join(new_file.split("/")[:-1])
            os.makedirs(new_dir, exist_ok=True)

            if "mel_min" in file or "mel_max" in file:
                os.system("cp {} {}".format(raw_file, new_file))
                continue

            with open(raw_file, "rb") as f:
                raw_feats = pickle.load(f)

            print("file: {}, #raw_feats = {}".format(file, len(raw_feats)))
            new_feats = [raw_feats[idx] for idx in chosen_idxs]
            with open(new_file, "wb") as f:
                pickle.dump(new_feats, f)

        # Utterance re-index
        news_utts = [utterances[idx] for idx in chosen_idxs]
        for i, utt in enumerate(news_utts):
            utt["Dataset"] = "vctksample"
            utt["index"] = i

        with open(os.path.join(sample_dir, "{}.json".format(split)), "w") as f:
            json.dump(news_utts, f, indent=4)
