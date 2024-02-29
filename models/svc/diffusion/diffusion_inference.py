# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler

from models.svc.base import SVCInference
from models.svc.diffusion.diffusion_inference_pipeline import DiffusionInferencePipeline
from models.svc.diffusion.diffusion_wrapper import DiffusionWrapper
from modules.encoder.condition_encoder import ConditionEncoder


class DiffusionInference(SVCInference):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        SVCInference.__init__(self, args, cfg, infer_type)

        settings = {
            **cfg.model.diffusion.scheduler_settings,
            **cfg.inference.diffusion.scheduler_settings,
        }
        settings.pop("num_inference_timesteps")

        if cfg.inference.diffusion.scheduler.lower() == "ddpm":
            self.scheduler = DDPMScheduler(**settings)
            self.logger.info("Using DDPM scheduler.")
        elif cfg.inference.diffusion.scheduler.lower() == "ddim":
            self.scheduler = DDIMScheduler(**settings)
            self.logger.info("Using DDIM scheduler.")
        elif cfg.inference.diffusion.scheduler.lower() == "pndm":
            self.scheduler = PNDMScheduler(**settings)
            self.logger.info("Using PNDM scheduler.")
        else:
            raise NotImplementedError(
                "Unsupported scheduler type: {}".format(
                    cfg.inference.diffusion.scheduler.lower()
                )
            )

        self.pipeline = DiffusionInferencePipeline(
            self.model[1],
            self.scheduler,
            args.diffusion_inference_steps,
        )

    def _build_model(self):
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = DiffusionWrapper(self.cfg)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        return model

    def _inference_each_batch(self, batch_data):
        device = self.accelerator.device
        for k, v in batch_data.items():
            batch_data[k] = v.to(device)

        conditioner = self.model[0](batch_data)
        noise = torch.randn_like(batch_data["mel"], device=device)
        y_pred = self.pipeline(noise, conditioner)
        return y_pred
