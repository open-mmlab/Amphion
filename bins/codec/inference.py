# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from argparse import ArgumentParser
import os

from models.codec.facodec.facodec_inference import FAcodecInference
from utils.util import load_config
import torch


def build_inference(args, cfg):
    supported_inference = {
        "FAcodec": FAcodecInference,
    }

    inference_class = supported_inference[cfg.model_type]
    inference = inference_class(args, cfg)
    return inference


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


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
        default=None,
        help="Acoustic model checkpoint directory. If a directory is given, "
        "search for the latest checkpoint dir in the directory. If a specific "
        "checkpoint dir is given, directly load the checkpoint.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source audio file",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to the reference audio file, passing an",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir for saving generated results",
    )
    return parser


def main():
    # Parse arguments
    parser = build_parser()
    args = parser.parse_args()
    print(args)

    # Parse config
    cfg = load_config(args.config)

    # CUDA settings
    cuda_relevant()

    # Build inference
    inferencer = build_inference(args, cfg)

    # Run inference
    _ = inferencer.inference(args.source, args.output_dir)

    # Run voice conversion
    if args.reference is not None:
        _ = inferencer.voice_conversion(args.source, args.reference, args.output_dir)


if __name__ == "__main__":
    main()
