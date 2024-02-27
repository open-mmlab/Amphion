# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from preprocessors import (
    m4singer,
    opencpop,
    svcc,
    pjs,
    popbutfy,
    opensinger,
    popcs,
    kising,
    csd,
    opera,
    nus48e,
    svcceval,
    vctk,
    vctksample,
    libritts,
    lijian,
    cdmusiceval,
    ljspeech,
    coco,
    cocoeval,
    customsvcdataset,
    vocalist,
    ljspeech_vocoder,
    librilight,
    hifitts,
)


def preprocess_dataset(
    dataset, dataset_path, output_path, cfg, task_type, is_custom_dataset=False
):
    """Call specific function to handle specific dataset
    Args:
        dataset (str): name of a dataset, e.g. opencpop, m4singer
        dataset_path (str): path to dataset
        output_path (str): path to store preprocessing result files
    """
    if is_custom_dataset:
        if task_type == "svc":
            customsvcdataset.main(output_path, dataset_path, dataset_name=dataset)
        else:
            raise NotImplementedError(
                "Custom dataset for {} task not implemented!".format(cfg.task_type)
            )

    if re.match("opencpop*", dataset):
        opencpop.main(dataset, output_path, dataset_path)
    if dataset == "m4singer":
        m4singer.main(output_path, dataset_path)
    if dataset == "svcc":
        svcc.main(output_path, dataset_path)
    if dataset == "pjs":
        pjs.main(output_path, dataset_path)
    if dataset == "popbutfy":
        popbutfy.main(output_path, dataset_path)
    if dataset == "opensinger":
        opensinger.main(output_path, dataset_path)
    if dataset == "popcs":
        popcs.main(output_path, dataset_path)
    if dataset == "kising":
        kising.main(output_path, dataset_path)
    if dataset == "csd":
        csd.main(output_path, dataset_path)
    if dataset == "opera":
        opera.main(output_path, dataset_path)
    if dataset == "nus48e":
        nus48e.main(output_path, dataset_path)
    if dataset == "vctk":
        vctk.main(output_path, dataset_path)
    if dataset == "svcceval":
        svcceval.main(output_path, dataset_path)
    if dataset == "libritts":
        libritts.main(output_path, dataset_path)
    if dataset == "lijian":
        lijian.main(output_path, dataset_path)
    if dataset == "cdmusiceval":
        cdmusiceval.main(output_path, dataset_path)
    if dataset == "LJSpeech":
        ljspeech.main(output_path, dataset_path, cfg)
    if dataset == "ljspeech":
        ljspeech_vocoder.main(output_path, dataset_path)
    if dataset == "coco":
        coco.main(output_path, dataset_path)
    if dataset == "cocoeval":
        cocoeval.main(output_path, dataset_path)
    if dataset == "vocalist":
        vocalist.main(output_path, dataset_path)
    if dataset == "librilight":
        librilight.main(output_path, dataset_path, cfg)
    if dataset == "hifitts":
        hifitts.main(output_path, dataset_path)


def prepare_align(dataset, dataset_path, cfg, output_path):
    """Call specific function to handle specific dataset

    Args:
        dataset (str): name of a dataset, e.g. ljspeech
        dataset_path (str): path to dataset
        output_path (str): path to store preprocessing result files
    """
    if dataset == "LJSpeech":
        ljspeech.prepare_align(dataset, dataset_path, cfg, output_path)
