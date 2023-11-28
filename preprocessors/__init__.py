# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
For source datasets' standard samples
"""

from collections import defaultdict
import os
import json

SPEECH_DATASETS = ["vctk", "vctksample"]

GOLDEN_TEST_SAMPLES = defaultdict(list)
GOLDEN_TEST_SAMPLES["m4singer"] = [
    "Alto-1_美错_0014",
    "Bass-1_十年_0008",
    "Soprano-2_同桌的你_0018",
    "Tenor-5_爱笑的眼睛_0010",
]
GOLDEN_TEST_SAMPLES["svcc"] = [
    # IDF1
    "IDF1_10030",
    "IDF1_10120",
    "IDF1_10140",
    # IDM1
    "IDM1_10001",
    "IDM1_10030",
    "IDM1_10120",
    # CDF1
    "CDF1_10030",
    "CDF1_10120",
    "CDF1_10140",
    # CDM1
    "CDM1_10001",
    "CDM1_10030",
    "CDM1_10120",
]
GOLDEN_TEST_SAMPLES["svcceval"] = [
    # SF1
    "SF1_30001",
    "SF1_30002",
    "SF1_30003",
    # SM1
    "SM1_30001",
    "SM1_30002",
    "SM1_30003",
]
GOLDEN_TEST_SAMPLES["popbutfy"] = [
    "Female1#you_are_my_sunshine_Professional#0",
    "Female4#Someone_Like_You_Professional#10",
    "Male2#Lemon_Tree_Professional#12",
    "Male5#can_you_feel_the_love_tonight_Professional#20",
]
GOLDEN_TEST_SAMPLES["opensinger"] = [
    "Man_0_大鱼_10",
    "Man_21_丑八怪_14",
    "Woman_39_mojito_22",
    "Woman_40_易燃易爆炸_12",
]
GOLDEN_TEST_SAMPLES["nus48e"] = [
    "ADIZ_read#01#0000",
    "MCUR_sing#10#0000",
    "JLEE_read#08#0001",
    "SAMF_sing#18#0001",
]
GOLDEN_TEST_SAMPLES["popcs"] = [
    "明天会更好_0004",
    "欧若拉_0005",
    "虫儿飞_0006",
    "隐形的翅膀_0008",
]
GOLDEN_TEST_SAMPLES["kising"] = [
    "421_0040",
    "424_0013",
    "431_0026",
]
GOLDEN_TEST_SAMPLES["csd"] = [
    "en_004a_0001",
    "en_042b_0006",
    "kr_013a_0006",
    "kr_045b_0004",
]
GOLDEN_TEST_SAMPLES["opera"] = [
    "fem_01#neg_1#0000",
    "fem_12#pos_3#0003",
    "male_02#neg_1#0002",
    "male_11#pos_2#0001",
]
GOLDEN_TEST_SAMPLES["lijian"] = [
    "058矜持_0000",
    "079绒花_0000",
    "120遥远的天空底下_0000",
]
GOLDEN_TEST_SAMPLES["cdmusiceval"] = ["陶喆_普通朋友", "蔡琴_给电影人的情书"]

GOLDEN_TRAIN_SAMPLES = defaultdict(list)


def get_golden_samples_indexes(
    dataset_name,
    dataset_dir=None,
    cfg=None,
    split=None,
    min_samples=5,
):
    """
    # Get Standard samples' indexes
    """
    if dataset_dir is None:
        assert cfg is not None
        dataset_dir = os.path.join(
            cfg.OUTPUT_PATH,
            "preprocess/{}_version".format(cfg.PREPROCESS_VERSION),
            dataset_name,
        )

    assert split is not None
    utt_file = os.path.join(dataset_dir, "{}.json".format(split))
    with open(utt_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if "train" in split:
        golden_samples = GOLDEN_TRAIN_SAMPLES[dataset_name]
    if "test" in split:
        golden_samples = GOLDEN_TEST_SAMPLES[dataset_name]

    res = []
    for idx, utt in enumerate(samples):
        if utt["Uid"] in golden_samples:
            res.append(idx)

        if dataset_name == "cdmusiceval":
            if "_".join(utt["Uid"].split("_")[:2]) in golden_samples:
                res.append(idx)

    if len(res) == 0:
        res = [i for i in range(min_samples)]

    return res


def get_specific_singer_indexes(dataset_dir, singer_name, split):
    utt_file = os.path.join(dataset_dir, "{}.json".format(split))
    with open(utt_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    res = []
    for idx, utt in enumerate(samples):
        if utt["Singer"] == singer_name:
            res.append(idx)

    assert len(res) != 0
    return res


def get_uids_and_wav_paths(
    cfg, dataset, dataset_type="train", only_specific_singer=None, return_singers=False
):
    dataset_dir = os.path.join(
        cfg.OUTPUT_PATH, "preprocess/{}_version".format(cfg.PREPROCESS_VERSION), dataset
    )
    dataset_file = os.path.join(
        dataset_dir, "{}.json".format(dataset_type.split("_")[-1])
    )
    with open(dataset_file, "r") as f:
        utterances = json.load(f)

    indexes = range(len(utterances))
    if "golden" in dataset_type:
        # golden_train or golden_test
        indexes = get_golden_samples_indexes(
            dataset, dataset_dir, split=dataset_type.split("_")[-1]
        )
    if only_specific_singer is not None:
        indexes = get_specific_singer_indexes(
            dataset_dir, only_specific_singer, dataset_type
        )

    uids = [utterances[i]["Uid"] for i in indexes]
    wav_paths = [utterances[i]["Path"] for i in indexes]
    singers = [utterances[i]["Singer"] for i in indexes]

    if not return_singers:
        return uids, wav_paths
    else:
        return uids, wav_paths, singers
