# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import random

from models.codec.coco.coco_dataset import CocoDataset

logger = logging.getLogger(__name__)


class VocosDataset(CocoDataset):
    def __init__(self, cfg):
        super(VocosDataset, self).__init__(cfg=cfg)

        self.longest_length = 3 * self.sample_rate  # 3 seconds

    def __getitem__(self, idx):
        single_features = super(VocosDataset, self).__getitem__(idx)

        wav = single_features["wav"]  # [T]

        if len(wav) > self.longest_length:
            start = random.randint(0, len(wav) - self.longest_length)
            wav = wav[start : start + self.longest_length]
        else:
            wav = np.pad(wav, (0, self.longest_length - len(wav)), mode="wrap")

        return {"wav": wav}
