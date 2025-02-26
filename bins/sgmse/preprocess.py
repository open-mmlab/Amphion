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


def preprocess(cfg):
    """Proprocess raw data of single or multiple datasets (in cfg.dataset)

    Args:
        cfg (dict): dictionary that stores configurations
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", help="json files for configurations."
    )
    parser.add_argument("--num_workers", type=int, default=int(cpu_count()))
    args = parser.parse_args()
    cfg = load_config(args.config)
    preprocess(cfg)


if __name__ == "__main__":
    main()
