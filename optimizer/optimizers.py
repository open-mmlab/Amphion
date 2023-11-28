# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``num_warmup`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    num_warmup: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """

    def __init__(self, optimizer, num_warmup):
        self.num_warmup = num_warmup
        self.base_lr = optimizer.param_groups[0]["lr"]
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = min(last_epoch ** (-0.5), last_epoch * self.num_warmup ** (-1.5))
        return [scale * self.base_lr]
