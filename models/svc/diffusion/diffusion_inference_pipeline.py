# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from diffusers import DiffusionPipeline


class DiffusionInferencePipeline(DiffusionPipeline):
    def __init__(self, network, scheduler, num_inference_timesteps=1000):
        super().__init__()

        self.register_modules(network=network, scheduler=scheduler)
        self.num_inference_timesteps = num_inference_timesteps

    @torch.inference_mode()
    def __call__(
        self,
        initial_noise: torch.Tensor,
        conditioner: torch.Tensor = None,
    ):
        r"""
        Args:
            initial_noise: The initial noise to be denoised.
            conditioner:The conditioner.
            n_inference_steps: The number of denoising steps. More denoising steps
                usually lead to a higher quality at the expense of slower inference.
        """

        mel = initial_noise
        batch_size = mel.size(0)
        self.scheduler.set_timesteps(self.num_inference_timesteps)

        for t in self.progress_bar(self.scheduler.timesteps):
            timestep = torch.full((batch_size,), t, device=mel.device, dtype=torch.long)

            # 1. predict noise model_output
            model_output = self.network(mel, timestep, conditioner)

            # 2. denoise, compute previous step: x_t -> x_t-1
            mel = self.scheduler.step(model_output, t, mel).prev_sample

            # 3. clamp
            mel = mel.clamp(-1.0, 1.0)

        return mel
