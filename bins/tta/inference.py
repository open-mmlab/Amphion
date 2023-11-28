# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from argparse import ArgumentParser
import os

from models.tta.ldm.audioldm_inference import AudioLDMInference
from utils.util import save_config, load_model_config, load_config
import numpy as np
import torch


def build_inference(args, cfg):
    supported_inference = {
        "AudioLDM": AudioLDMInference,
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
        "--text",
        help="Text to be synthesized",
        type=str,
        default="Text to be synthesized.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
    )
    parser.add_argument(
        "--vocoder_path", type=str, help="Checkpoint path of the vocoder"
    )
    parser.add_argument(
        "--vocoder_config_path", type=str, help="Config path of the vocoder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir for saving generated results",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="The total number of denosing steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="The scale of classifer free guidance",
    )
    parser.add_argument("--local_rank", default=-1, type=int)
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
