# ruff: noqa: E741

from models.vocoders.gan.generator.hifigan import HiFiGAN

import os

import torch
from omegaconf import OmegaConf


# hifigan vctk-v1
def load_hifigan(ckpt_path):
    config = OmegaConf.load(os.path.join(ckpt_path, "config.json"))
    ckpt = torch.load(os.path.join(ckpt_path, "generator_v1"))

    vocoder = HiFiGAN(
        OmegaConf.create(
            {
                "model": {"hifigan": config},
                "preprocess": {"n_mel": config.num_mels},
            }
        )
    )
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder, config
