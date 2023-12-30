# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from models.vocoders.gan.gan_vocoder_trainer import GANVocoderTrainer
from models.vocoders.diffusion.diffusion_vocoder_trainer import DiffusionVocoderTrainer

from utils.util import load_config


def build_trainer(args, cfg):
    supported_trainer = {
        "GANVocoder": GANVocoderTrainer,
        "DiffusionVocoder": DiffusionVocoderTrainer,
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


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

    # Data Augmentation
    if cfg.preprocess.data_augment:
        new_datasets_list = []
        for dataset in cfg.preprocess.data_augment:
            new_datasets = [
                # f"{dataset}_pitch_shift",
                # f"{dataset}_formant_shift",
                f"{dataset}_equalizer",
                f"{dataset}_time_stretch",
            ]
            new_datasets_list.extend(new_datasets)
        cfg.dataset.extend(new_datasets_list)

    # CUDA settings
    cuda_relevant()

    # Build trainer
    trainer = build_trainer(args, cfg)

    trainer.train_loop()


if __name__ == "__main__":
    main()
