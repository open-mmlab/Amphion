# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def log_dur_loss(dur_pred_log, dur_target, mask, loss_type="l1"):
    # dur_pred_log: (B, N)
    # dur_target: (B, N)
    # mask: (B, N) mask is 0
    dur_target_log = torch.log(1 + dur_target)
    if loss_type == "l1":
        loss = F.l1_loss(
            dur_pred_log, dur_target_log, reduction="none"
        ).float() * mask.to(dur_target.dtype)
    elif loss_type == "l2":
        loss = F.mse_loss(
            dur_pred_log, dur_target_log, reduction="none"
        ).float() * mask.to(dur_target.dtype)
    else:
        raise NotImplementedError()
    loss = loss.sum() / (mask.to(dur_target.dtype).sum())
    return loss


def log_pitch_loss(pitch_pred_log, pitch_target, mask, loss_type="l1"):
    pitch_target_log = torch.log(pitch_target)
    if loss_type == "l1":
        loss = F.l1_loss(
            pitch_pred_log, pitch_target_log, reduction="none"
        ).float() * mask.to(pitch_target.dtype)
    elif loss_type == "l2":
        loss = F.mse_loss(
            pitch_pred_log, pitch_target_log, reduction="none"
        ).float() * mask.to(pitch_target.dtype)
    else:
        raise NotImplementedError()
    loss = loss.sum() / (mask.to(pitch_target.dtype).sum() + 1e-8)
    return loss


def diff_loss(pred, target, mask, loss_type="l1"):
    # pred: (B, d, T)
    # target: (B, d, T)
    # mask: (B, T)
    if loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(1)
        )
    elif loss_type == "l2":
        loss = F.mse_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(1)
        )
    else:
        raise NotImplementedError()
    loss = (torch.mean(loss, dim=1)).sum() / (mask.to(pred.dtype).sum())
    return loss


def diff_ce_loss(pred_dist, gt_indices, mask):
    # pred_dist: (nq, B, T, 1024)
    # gt_indices: (nq, B, T)
    pred_dist = pred_dist.permute(1, 3, 0, 2)  # (B, 1024, nq, T)
    gt_indices = gt_indices.permute(1, 0, 2).long()  # (B, nq, T)
    loss = F.cross_entropy(
        pred_dist, gt_indices, reduction="none"
    ).float()  # (B, nq, T)
    loss = loss * mask.to(loss.dtype).unsqueeze(1)
    loss = (torch.mean(loss, dim=1)).sum() / (mask.to(loss.dtype).sum())
    return loss
