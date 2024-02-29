# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from diffusers import DDPMScheduler

from models.svc.base import SVCTrainer
from modules.encoder.condition_encoder import ConditionEncoder
from .diffusion_wrapper import DiffusionWrapper


class DiffusionTrainer(SVCTrainer):
    r"""The base trainer for all diffusion models. It inherits from SVCTrainer and
    implements ``_build_model`` and ``_forward_step`` methods.
    """

    def __init__(self, args=None, cfg=None):
        SVCTrainer.__init__(self, args, cfg)

        # Only for SVC tasks using diffusion
        self.noise_scheduler = DDPMScheduler(
            **self.cfg.model.diffusion.scheduler_settings,
        )
        self.diffusion_timesteps = (
            self.cfg.model.diffusion.scheduler_settings.num_train_timesteps
        )

    ### Following are methods only for diffusion models ###
    def _build_model(self):
        r"""Build the model for training. This function is called in ``__init__`` function."""

        # TODO: sort out the config
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = DiffusionWrapper(self.cfg)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])

        num_of_params_encoder = self.count_parameters(self.condition_encoder)
        num_of_params_am = self.count_parameters(self.acoustic_mapper)
        num_of_params = num_of_params_encoder + num_of_params_am
        log = "Diffusion Model's Parameters: #Encoder is {:.2f}M, #Diffusion is {:.2f}M. The total is {:.2f}M".format(
            num_of_params_encoder / 1e6, num_of_params_am / 1e6, num_of_params / 1e6
        )
        self.logger.info(log)

        return model

    def count_parameters(self, model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def _check_nan(self, batch, loss, y_pred, y_gt):
        if torch.any(torch.isnan(loss)):
            for k, v in batch.items():
                self.logger.info(k)
                self.logger.info(v)

            super()._check_nan(loss, y_pred, y_gt)

    def _forward_step(self, batch):
        r"""Forward step for training and inference. This function is called
        in ``_train_step`` & ``_test_step`` function.
        """
        device = self.accelerator.device

        if self.online_features_extraction:
            # On-the-fly features extraction
            batch = self._extract_svc_features(batch)

            # To debug
            # for k, v in batch.items():
            #     print(k, v.shape, v)
            # exit()

        mel_input = batch["mel"]
        noise = torch.randn_like(mel_input, device=device, dtype=torch.float32)
        batch_size = mel_input.size(0)
        timesteps = torch.randint(
            0,
            self.diffusion_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        noisy_mel = self.noise_scheduler.add_noise(mel_input, noise, timesteps)
        conditioner = self.condition_encoder(batch)

        y_pred = self.acoustic_mapper(noisy_mel, timesteps, conditioner)

        loss = self._compute_loss(self.criterion, y_pred, noise, batch["mask"])
        self._check_nan(batch, loss, y_pred, noise)

        return loss
