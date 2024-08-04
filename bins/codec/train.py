# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from models.codec.facodec.facodec_trainer import FAcodecTrainer

from utils.util import load_config


def build_trainer(args, cfg):
    supported_trainer = {
        "FAcodec": FAcodecTrainer,
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        help="resume for continue to train, finetune for finetuning",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint to resume",
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    # CUDA settings
    cuda_relevant()

    # Build trainer
    trainer = build_trainer(args, cfg)

    trainer.train_loop()


if __name__ == "__main__":
    main()
