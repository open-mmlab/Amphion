# -*- coding: utf-8 -*-

# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Adversarial loss modules."""

import torch
import torch.nn.functional as F


class GeneratorAdversarialLoss(torch.nn.Module):
    """Generator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators=True,
        loss_type="mse",
    ):
        """Initialize GeneratorAversarialLoss module."""
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.criterion = self._mse_loss
        else:
            self.criterion = self._hinge_loss

    def forward(self, outputs):
        """Calcualate generator adversarial loss.

        Args:
            outputs (Tensor or list): Discriminator outputs or list of
                discriminator outputs.

        Returns:
            Tensor: Generator adversarial loss value.

        """
        if isinstance(outputs, (tuple, list)):
            adv_loss = 0.0
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_ = outputs_[-1]
                adv_loss += self.criterion(outputs_)
            if self.average_by_discriminators:
                adv_loss /= i + 1
        else:
            adv_loss = self.criterion(outputs)

        return adv_loss

    def _mse_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _hinge_loss(self, x):
        return -x.mean()


class DiscriminatorAdversarialLoss(torch.nn.Module):
    """Discriminator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators=True,
        loss_type="mse",
    ):
        """Initialize DiscriminatorAversarialLoss module."""
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(self, outputs_hat, outputs):
        """Calcualate discriminator adversarial loss.

        Args:
            outputs_hat (Tensor or list): Discriminator outputs or list of
                discriminator outputs calculated from generator outputs.
            outputs (Tensor or list): Discriminator outputs or list of
                discriminator outputs calculated from groundtruth.

        Returns:
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.

        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)

        return real_loss, fake_loss

    def _mse_real_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x):
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x):
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x):
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))
