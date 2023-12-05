# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from models.base.base_dataset import ( 
    BaseDataset,
    BaseCollator,
    BaseTestDataset,
    BaseTestCollator
)
    
from processors.content_extractor import (
    ContentvecExtractor,
    WenetExtractor,
    WhisperExtractor,
)


class TTSDataset(BaseDataset):
    def __init__(self, args, cfg, is_valid=False):
        super().__init__(args, cfg, is_valid)
    

    def __getitem__(self, index):
        single_feature = super().__getitem__(index)
        return single_feature

    def __len__(self):
        
        return super().__len__()


class TTSCollator(BaseCollator):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, batch):
        parsed_batch_features = super().__call__(batch)
        return parsed_batch_features


class TTSTestDataset(BaseTestDataset):
    def __init__(self, args, cfg):
        self.cfg = cfg
        
        # inference from test list file
        if args.test_list_file is not None:
            # construst metadata
            self.metadata = []
            
            with open(args.test_list_file, "r") as fin:
                for idx, line in enumerate(fin.readlines()):
                    utt_info = {}
                    
                    utt_info["Dataset"] = "test"
                    utt_info["Text"] = line.strip()
                    utt_info["Uid"] = str(idx)
                    self.metadata.append(utt_info)
                    
        else:
            assert args.testing_set
            self.metafile_path = os.path.join(
                cfg.preprocess.processed_dir,
                args.dataset,
                "{}.json".format(args.testing_set),
            )
            self.metadata = self.get_metadata()


    def __getitem__(self, index):
        single_feature = {}

        return single_feature

    def __len__(self):
        return len(self.metadata)


class TTSTestCollator(BaseTestCollator):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, T, n_mels]
        # frame_pitch, frame_energy: [1, T]
        # target_len: [1]
        # spk_id: [b, 1]
        # mask: [b, T, 1]

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "phone_len":
                packed_batch_features["phone_len"] = torch.LongTensor(
                    [b["phone_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["phone_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["phn_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "audio_len":
                packed_batch_features["audio_len"] = torch.LongTensor(
                    [b["audio_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["audio_len"], 1), dtype=torch.long) for b in batch
                ]
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )
        return packed_batch_features
