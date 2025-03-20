# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from argparse import ArgumentParser
import os

from models.sgmse.dereverberation.dereverberation_inference import (
    DereverberationInference,
)
from utils.util import save_config, load_model_config, load_config
import numpy as np
import torch


def build_inference(args, cfg):
    supported_inference = {
        "dereverberation": DereverberationInference,
    }

    inference_class = supported_inference[cfg.model_type]
    inference = inference_class(args, cfg)
    return inference


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON/YAML file for configurations.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing the test data (must have subdirectory noisy/)",
    )
    parser.add_argument(
        "--corrector_steps", type=int, default=1, help="Number of corrector steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir for saving generated results",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=0.33,
        help="SNR value for (annealed) Langevin dynmaics.",
    )
    parser.add_argument("--N", type=int, default=50, help="Number of reverse steps")
    parser.add_argument("--local_rank", default=0, type=int)
    return parser


def main():
    # Parse arguments
    args = build_parser().parse_args()
    # args, infer_type = formulate_parser(args)

    # Parse config
    cfg = load_config(args.config)
    if torch.cuda.is_available():
        args.local_rank = torch.device("cuda")
    else:
        args.local_rank = torch.device("cpu")
    print("args: ", args)

    # Build inference
    inferencer = build_inference(args, cfg)

    # Run inference
    inferencer.inference()


if __name__ == "__main__":
    main()
