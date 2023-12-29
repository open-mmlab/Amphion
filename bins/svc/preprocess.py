# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faulthandler

faulthandler.enable()

import os
import argparse
import json
from multiprocessing import cpu_count


from utils.util import load_config
from preprocessors.processor import preprocess_dataset
from preprocessors.metadata import cal_metadata
from processors import acoustic_extractor, content_extractor, data_augment


def extract_acoustic_features(dataset, output_path, cfg, n_workers=1):
    """Extract acoustic features of utterances in the dataset

    Args:
        dataset (str): name of dataset, e.g. opencpop
        output_path (str): directory that stores train, test and feature files of datasets
        cfg (dict): dictionary that stores configurations
        n_workers (int, optional): num of processes to extract features in parallel. Defaults to 1.
    """
    types = ["train", "test"] if "eval" not in dataset else ["test"]
    metadata = []
    dataset_output = os.path.join(output_path, dataset)

    for dataset_type in types:
        dataset_file = os.path.join(dataset_output, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

        # acoustic_extractor.extract_utt_acoustic_features_parallel(
        #     metadata, dataset_output, cfg, n_workers=n_workers
        # )
    acoustic_extractor.extract_utt_acoustic_features_serial(
        metadata, dataset_output, cfg
    )


def extract_content_features(dataset, output_path, cfg, num_workers=1):
    """Extract content features of utterances in the dataset

    Args:
        dataset (str): name of dataset, e.g. opencpop
        output_path (str): directory that stores train, test and feature files of datasets
        cfg (dict): dictionary that stores configurations
    """
    types = ["train", "test"] if "eval" not in dataset else ["test"]
    metadata = []
    for dataset_type in types:
        dataset_output = os.path.join(output_path, dataset)
        dataset_file = os.path.join(dataset_output, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

    content_extractor.extract_utt_content_features_dataloader(
        cfg, metadata, num_workers
    )


def preprocess(cfg, args):
    """Proprocess raw data of single or multiple datasets (in cfg.dataset)

    Args:
        cfg (dict): dictionary that stores configurations
        args (ArgumentParser): specify the configuration file and num_workers
    """
    # Specify the output root path to save the processed data
    output_path = cfg.preprocess.processed_dir
    os.makedirs(output_path, exist_ok=True)

    ## Split train and test sets
    for dataset in cfg.dataset:
        print("Preprocess {}...".format(dataset))
        preprocess_dataset(
            dataset,
            cfg.dataset_path[dataset],
            output_path,
            cfg.preprocess,
            cfg.task_type,
            is_custom_dataset=dataset in cfg.use_custom_dataset,
        )

    # Data augmentation: create new wav files with pitch shift, formant shift, equalizer, time stretch
    try:
        assert isinstance(
            cfg.preprocess.data_augment, list
        ), "Please provide a list of datasets need to be augmented."
        if len(cfg.preprocess.data_augment) > 0:
            new_datasets_list = []
            for dataset in cfg.preprocess.data_augment:
                new_datasets = data_augment.augment_dataset(cfg, dataset)
                new_datasets_list.extend(new_datasets)
            cfg.dataset.extend(new_datasets_list)
            print("Augmentation datasets: ", cfg.dataset)
    except:
        print("No Data Augmentation.")

    # Dump metadata of datasets (singers, train/test durations, etc.)
    cal_metadata(cfg)

    ## Prepare the acoustic features
    for dataset in cfg.dataset:
        # Skip augmented datasets which do not need to extract acoustic features
        # We will copy acoustic features from the original dataset later
        if (
            "pitch_shift" in dataset
            or "formant_shift" in dataset
            or "equalizer" in dataset in dataset
        ):
            continue
        print(
            "Extracting acoustic features for {} using {} workers ...".format(
                dataset, args.num_workers
            )
        )
        extract_acoustic_features(dataset, output_path, cfg, args.num_workers)
        # Calculate the statistics of acoustic features
        if cfg.preprocess.mel_min_max_norm:
            acoustic_extractor.cal_mel_min_max(dataset, output_path, cfg)

        if cfg.preprocess.extract_pitch:
            acoustic_extractor.cal_pitch_statistics_svc(dataset, output_path, cfg)

    # Copy acoustic features for augmented datasets by creating soft-links
    for dataset in cfg.dataset:
        if "pitch_shift" in dataset:
            src_dataset = dataset.replace("_pitch_shift", "")
            src_dataset_dir = os.path.join(output_path, src_dataset)
        elif "formant_shift" in dataset:
            src_dataset = dataset.replace("_formant_shift", "")
            src_dataset_dir = os.path.join(output_path, src_dataset)
        elif "equalizer" in dataset:
            src_dataset = dataset.replace("_equalizer", "")
            src_dataset_dir = os.path.join(output_path, src_dataset)
        else:
            continue
        dataset_dir = os.path.join(output_path, dataset)
        metadata = []
        for split in ["train", "test"] if not "eval" in dataset else ["test"]:
            metadata_file_path = os.path.join(src_dataset_dir, "{}.json".format(split))
            with open(metadata_file_path, "r") as f:
                metadata.extend(json.load(f))
        print("Copying acoustic features for {}...".format(dataset))
        acoustic_extractor.copy_acoustic_features(
            metadata, dataset_dir, src_dataset_dir, cfg
        )
        if cfg.preprocess.mel_min_max_norm:
            acoustic_extractor.cal_mel_min_max(dataset, output_path, cfg)

        if cfg.preprocess.extract_pitch:
            acoustic_extractor.cal_pitch_statistics(dataset, output_path, cfg)

    # Prepare the content features
    for dataset in cfg.dataset:
        print("Extracting content features for {}...".format(dataset))
        extract_content_features(dataset, output_path, cfg, args.num_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", help="json files for configurations."
    )
    parser.add_argument("--num_workers", type=int, default=int(cpu_count()))
    parser.add_argument("--prepare_alignment", type=bool, default=False)

    args = parser.parse_args()
    cfg = load_config(args.config)

    preprocess(cfg, args)


if __name__ == "__main__":
    main()
