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


# Train: male 20 hours, female 10 hours
TRAIN_MALE_MAX_SECONDS = 20 * 3600
TRAIN_FEMALE_MAX_SECONDS = 10 * 3600
TEST_MAX_NUM_EVERY_PERSON = 5


def select_sample_idxs():
    chosen_speakers = get_chosen_speakers()

    with open(os.path.join(vctk_dir, "train.json"), "r") as f:
        raw_train = json.load(f)
    with open(os.path.join(vctk_dir, "test.json"), "r") as f:
        raw_test = json.load(f)

    train_idxs, test_idxs = [], []

    # =========== Test ===========
    test_nums = defaultdict(int)
    for utt in tqdm(raw_train):
        idx = utt["index"]
        singer = utt["Singer"]

        if singer in chosen_speakers and test_nums[singer] < TEST_MAX_NUM_EVERY_PERSON:
            test_nums[singer] += 1
            test_idxs.append("train_{}".format(idx))

    for utt in tqdm(raw_test):
        idx = utt["index"]
        singer = utt["Singer"]

        if singer in chosen_speakers and test_nums[singer] < TEST_MAX_NUM_EVERY_PERSON:
            test_nums[singer] += 1
            test_idxs.append("test_{}".format(idx))

    # =========== Train ===========
    for utt in tqdm(raw_train):
        idx = utt["index"]
        singer = utt["Singer"]

        if singer in chosen_speakers and "train_{}".format(idx) not in test_idxs:
            train_idxs.append("train_{}".format(idx))

    for utt in tqdm(raw_test):
        idx = utt["index"]
        singer = utt["Singer"]

        if singer in chosen_speakers and "test_{}".format(idx) not in test_idxs:
            train_idxs.append("test_{}".format(idx))

    train_idxs.sort()
    test_idxs.sort()
    return train_idxs, test_idxs, raw_train, raw_test


def statistics_of_speakers():
    speaker2time = defaultdict(float)
    sex2time = defaultdict(float)

    with open(os.path.join(vctk_dir, "train.json"), "r") as f:
        train = json.load(f)
    with open(os.path.join(vctk_dir, "test.json"), "r") as f:
        test = json.load(f)

    for utt in train + test:
        # minutes
        speaker2time[utt["Singer"]] += utt["Duration"]
        # hours
        sex2time[utt["Singer"].split("_")[0]] += utt["Duration"]

    print(
        "Female: {:.2f} hours, Male: {:.2f} hours.\n".format(
            sex2time["female"] / 3600, sex2time["male"] / 3600
        )
    )

    speaker2time = sorted(speaker2time.items(), key=lambda x: x[-1], reverse=True)
    for singer, seconds in speaker2time:
        print("{}\t{:.2f} mins".format(singer, seconds / 60))

    return speaker2time


def get_chosen_speakers():
    speaker2time = statistics_of_speakers()

    chosen_time = defaultdict(float)
    chosen_speaker = defaultdict(list)
    train_constrait = {
        "male": TRAIN_MALE_MAX_SECONDS,
        "female": TRAIN_FEMALE_MAX_SECONDS,
    }

    for speaker, seconds in speaker2time:
        sex = speaker.split("_")[0]
        if chosen_time[sex] < train_constrait[sex]:
            chosen_time[sex] += seconds
            chosen_speaker[sex].append(speaker)

    speaker2time = dict(speaker2time)
    chosen_speaker = chosen_speaker["male"] + chosen_speaker["female"]
    print("\n#Chosen speakers = {}".format(len(chosen_speaker)))
    for spk in chosen_speaker:
        print("{}\t{:.2f} mins".format(spk, speaker2time[spk] / 60))

    return chosen_speaker


if __name__ == "__main__":
    root_path = ""
    vctk_dir = os.path.join(root_path, "vctk")
    fewspeaker_dir = os.path.join(root_path, "vctkfewspeaker")
    os.makedirs(fewspeaker_dir, exist_ok=True)

    train_idxs, test_idxs, raw_train, raw_test = select_sample_idxs()
    print("#Train = {}, #Test = {}".format(len(train_idxs), len(test_idxs)))

    # There are no data leakage
    assert len(set(train_idxs).intersection(set(test_idxs))) == 0
    for idx in train_idxs + test_idxs:
        # No test data of raw vctk
        assert "test_" not in idx

    for split, chosen_idxs in zip(["train", "test"], [train_idxs, test_idxs]):
        print("{}: #chosen idx = {}\n".format(split, len(chosen_idxs)))

        # Select features
        feat_files = glob.glob("**/train.pkl", root_dir=vctk_dir, recursive=True)
        for file in tqdm(feat_files):
            raw_file = os.path.join(vctk_dir, file)
            new_file = os.path.join(
                fewspeaker_dir, file.replace("train.pkl", "{}.pkl".format(split))
            )

            new_dir = "/".join(new_file.split("/")[:-1])
            os.makedirs(new_dir, exist_ok=True)

            if "mel_min" in file or "mel_max" in file:
                os.system("cp {} {}".format(raw_file, new_file))
                continue

            with open(raw_file, "rb") as f:
                raw_feats = pickle.load(f)

            print("file: {}, #raw_feats = {}".format(file, len(raw_feats)))
            new_feats = []
            for idx in chosen_idxs:
                chosen_split_is_train, raw_idx = idx.split("_")
                assert chosen_split_is_train == "train"
                new_feats.append(raw_feats[int(raw_idx)])

            with open(new_file, "wb") as f:
                pickle.dump(new_feats, f)
            print("New file: {}, #new_feats = {}".format(new_file, len(new_feats)))

        # Utterance re-index
        news_utts = [raw_train[int(idx.split("_")[-1])] for idx in chosen_idxs]
        for i, utt in enumerate(news_utts):
            utt["Dataset"] = "vctkfewsinger"
            utt["index"] = i

        with open(os.path.join(fewspeaker_dir, "{}.json".format(split)), "w") as f:
            json.dump(news_utts, f, indent=4)
