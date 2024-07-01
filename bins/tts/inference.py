# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from argparse import ArgumentParser
import os

from models.tts.fastspeech2.fs2_inference import FastSpeech2Inference
from models.tts.vits.vits_inference import VitsInference
from models.tts.valle.valle_inference import VALLEInference
from models.tts.naturalspeech2.ns2_inference import NS2Inference
from models.tts.jets.jets_inference import JetsInference
from utils.util import load_config
import torch


def build_inference(args, cfg):
    supported_inference = {
        "FastSpeech2": FastSpeech2Inference,
        "VITS": VitsInference,
        "VALLE": VALLEInference,
        "NaturalSpeech2": NS2Inference,
        "Jets": JetsInference,
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
        "--dataset",
        type=str,
        help="convert from the source data",
        default=None,
    )
    parser.add_argument(
        "--testing_set",
        type=str,
        help="train, test, golden_test",
        default="test",
    )
    parser.add_argument(
        "--test_list_file",
        type=str,
        help="convert from the test list file",
        default=None,
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default=None,
        help="speaker name for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--text",
        help="Text to be synthesized.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--vocoder_dir",
        type=str,
        default=None,
        help="Vocoder checkpoint directory. Searching behavior is the same as "
        "the acoustics one.",
    )
    parser.add_argument(
        "--acoustics_dir",
        type=str,
        default=None,
        help="Acoustic model checkpoint directory. If a directory is given, "
        "search for the latest checkpoint dir in the directory. If a specific "
        "checkpoint dir is given, directly load the checkpoint.",
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
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        help="Logging level. Default: warning",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
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
    VALLEInference.add_arguments(parser)
    NS2Inference.add_arguments(parser)
    args = parser.parse_args()
    print(args)

    # Parse config
    cfg = load_config(args.config)

    # CUDA settings
    cuda_relevant()

    # Build inference
    inferencer = build_inference(args, cfg)

    # Run inference
    inferencer.inference()


if __name__ == "__main__":
    main()
