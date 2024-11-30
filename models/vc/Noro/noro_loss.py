# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class ConstractiveSpeakerLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ConstractiveSpeakerLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, speaker_ids):
        # x : B, H
        # speaker_ids: B 3 4 3
        speaker_ids = speaker_ids.reshape(-1)
        speaker_ids_expand = torch.zeros(len(speaker_ids), len(speaker_ids)).to(
            speaker_ids.device
        )
        speaker_ids_expand = (speaker_ids.view(-1, 1) == speaker_ids).float()
        x_t = x.transpose(0, 1)  # B, C --> C,B
        logits = (x @ x_t) / self.temperature  # B, H * H, B --> B, B
        targets = F.softmax(speaker_ids_expand / self.temperature, dim=-1)
        loss = cross_entropy_loss(logits, targets, reduction="none")
        return loss.mean()


def diff_loss(pred, target, mask, loss_type="l1"):
    # pred: (B, T, d)
    # target: (B, T, d)
    # mask: (B, T)
    if loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(-1)
        )
    elif loss_type == "l2":
        loss = F.mse_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(-1)
        )
    else:
        raise NotImplementedError()
    loss = (torch.mean(loss, dim=-1)).sum() / (mask.to(pred.dtype).sum())
    return loss
