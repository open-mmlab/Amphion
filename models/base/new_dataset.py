# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from abc import abstractmethod
from pathlib import Path

import json5
import torch
import yaml


# TODO: for training and validating
class BaseDataset(torch.utils.data.Dataset):
    r"""Base dataset for training and validating."""

    def __init__(self, args, cfg, is_valid=False):
        pass


class BaseTestDataset(torch.utils.data.Dataset):
    r"""Test dataset for inference."""

    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        assert infer_type in ["from_dataset", "from_file"]

        self.args = args
        self.cfg = cfg
        self.infer_type = infer_type

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        path = Path(self.args.source)
        if path.suffix == ".json" or path.suffix == ".jsonc":
            metadata = json5.load(open(self.args.source, "r"))
        elif path.suffix == ".yaml" or path.suffix == ".yml":
            metadata = yaml.full_load(open(self.args.source, "r"))
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return metadata
