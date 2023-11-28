# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch

from models.tta.autoencoder.autoencoder_trainer import AutoencoderKLTrainer
from models.tta.ldm.audioldm_trainer import AudioLDMTrainer
from utils.util import load_config


def build_trainer(args, cfg):
    supported_trainer = {
        "AutoencoderKL": AutoencoderKLTrainer,
        "AudioLDM": AudioLDMTrainer,
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        # action="store_true",
        help="The model name to restore",
    )
    parser.add_argument(
        "--log_level", default="info", help="logging level (info, debug, warning)"
    )
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.exp_name = args.exp_name

    # Model saving dir
    args.log_dir = os.path.join(cfg.log_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=True)

    if not cfg.train.ddp:
        args.local_rank = torch.device("cuda")

    # Build trainer
    trainer = build_trainer(args, cfg)

    # Restore models
    if args.resume:
        trainer.restore()
    trainer.train()


if __name__ == "__main__":
    main()
