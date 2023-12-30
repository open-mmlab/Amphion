# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch

from models.vocoders.vocoder_inference import VocoderInference
from utils.util import load_config


def build_inference(args, cfg, infer_type="infer_from_dataset"):
    supported_inference = {
        "GANVocoder": VocoderInference,
        "DiffusionVocoder": VocoderInference,
    }

    inference_class = supported_inference[cfg.model_type]
    return inference_class(args, cfg, infer_type)


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
    r"""Build argument parser for inference.py.
    Anything else should be put in an extra config YAML file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON/YAML file for configurations.",
    )
    parser.add_argument(
        "--infer_mode",
        type=str,
        required=None,
    )
    parser.add_argument(
        "--infer_datasets",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--feature_folder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--audio_folder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vocoder_dir",
        type=str,
        required=True,
        help="Vocoder checkpoint directory. Searching behavior is the same as "
        "the acoustics one.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result",
        help="Output directory. Default: ./result",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        help="Logging level. Default: warning",
    )
    parser.add_argument(
        "--keep_cache",
        action="store_true",
        default=False,
        help="Keep cache files. Only applicable to inference from files.",
    )
    return parser


def main():
    # Parse arguments
    args = build_parser().parse_args()

    # Parse config
    cfg = load_config(args.config)

    # CUDA settings
    cuda_relevant()

    # Build inference
    trainer = build_inference(args, cfg, args.infer_mode)

    # Run inference
    trainer.inference()


if __name__ == "__main__":
    main()
