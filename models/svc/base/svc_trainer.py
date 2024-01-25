# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import torch
import torch.nn as nn

from models.base.new_trainer import BaseTrainer
from models.svc.base.svc_dataset import SVCCollator, SVCDataset


class SVCTrainer(BaseTrainer):
    r"""The base trainer for all SVC models. It inherits from BaseTrainer and implements
    ``build_criterion``, ``_build_dataset`` and ``_build_singer_lut`` methods. You can inherit from this
    class, and implement ``_build_model``, ``_forward_step``.
    """

    def __init__(self, args=None, cfg=None):
        self.args = args
        self.cfg = cfg

        self._init_accelerator()

        # Only for SVC tasks
        with self.accelerator.main_process_first():
            self.singers = self._build_singer_lut()

        # Super init
        BaseTrainer.__init__(self, args, cfg)

        # Only for SVC tasks
        self.task_type = "SVC"
        self.logger.info("Task type: {}".format(self.task_type))

    ### Following are methods only for SVC tasks ###
    # TODO: LEGACY CODE, NEED TO BE REFACTORED
    def _build_dataset(self):
        return SVCDataset, SVCCollator

    @staticmethod
    def _build_criterion():
        criterion = nn.MSELoss(reduction="none")
        return criterion

    @staticmethod
    def _compute_loss(criterion, y_pred, y_gt, loss_mask):
        """
        Args:
            criterion: MSELoss(reduction='none')
            y_pred, y_gt: (bs, seq_len, D)
            loss_mask: (bs, seq_len, 1)
        Returns:
            loss: Tensor of shape []
        """

        # (bs, seq_len, D)
        loss = criterion(y_pred, y_gt)
        # expand loss_mask to (bs, seq_len, D)
        loss_mask = loss_mask.repeat(1, 1, loss.shape[-1])

        loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)
        return loss

    def _save_auxiliary_states(self):
        """
        To save the singer's look-up table in the checkpoint saving path
        """
        with open(
            os.path.join(self.tmp_checkpoint_save_path, self.cfg.preprocess.spk2id),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.singers, f, indent=4, ensure_ascii=False)

    def _build_singer_lut(self):
        resumed_singer_path = None
        if self.args.resume_from_ckpt_path and self.args.resume_from_ckpt_path != "":
            resumed_singer_path = os.path.join(
                self.args.resume_from_ckpt_path, self.cfg.preprocess.spk2id
            )
        if os.path.exists(os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)):
            resumed_singer_path = os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)

        if resumed_singer_path:
            with open(resumed_singer_path, "r") as f:
                singers = json.load(f)
        else:
            singers = dict()

        for dataset in self.cfg.dataset:
            singer_lut_path = os.path.join(
                self.cfg.preprocess.processed_dir, dataset, self.cfg.preprocess.spk2id
            )
            with open(singer_lut_path, "r") as singer_lut_path:
                singer_lut = json.load(singer_lut_path)
            for singer in singer_lut.keys():
                if singer not in singers:
                    singers[singer] = len(singers)

        with open(
            os.path.join(self.exp_dir, self.cfg.preprocess.spk2id), "w"
        ) as singer_file:
            json.dump(singers, singer_file, indent=4, ensure_ascii=False)
        print(
            "singers have been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            )
        )
        return singers
