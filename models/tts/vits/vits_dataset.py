# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from models.base.base_dataset import (
    BaseCollator,
    BaseDataset,
    BaseTestDataset,
    BaseTestCollator,
)
from text import text_to_sequence
from text.g2p import preprocess_english, read_lexicon


class VITSDataset(BaseDataset):
    def __init__(self, cfg, dataset, is_valid):
        BaseDataset.__init__(self, cfg, dataset, is_valid=is_valid)

    def __getitem__(self, index):
        single_feature = BaseDataset.__getitem__(self, index)
        return single_feature

    def __len__(self):
        return len(self.metadata)


class VITSCollator(BaseCollator):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)

    def __call__(self, batch):
        parsed_batch_features = BaseCollator.__call__(self, batch)
        return parsed_batch_features


class VITSTestDataset(BaseTestDataset):
    def __init__(self, args, cfg, target_singer=None):
        self.cfg = cfg
        if args.test_list_file is not None:
            self.metadata = []
            if cfg.preprocess.use_phone:
                lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)

            with open(args.test_list_file, "r") as fin:
                for idx, line in enumerate(fin.readlines()):
                    utt_info = {}
                    utt_info["Dataset"] = "null"
                    utt_info["Text"] = line.strip()
                    utt_info["Uid"] = str(idx)
                    if cfg.preprocess.use_phone:
                        # convert text to phone sequence
                        phone = preprocess_english(utt_info["Text"], lexicon)
                        utt_info["Phone"] = phone
                    self.metadata.append(utt_info)
        else:
            assert args.testing_set
            self.metafile_path = os.path.join(
                cfg.preprocess.processed_dir,
                args.dataset,
                "{}.json".format(args.testing_set),
            )
            self.metadata = self.get_metadata()

        if cfg.preprocess.use_spkid:
            spk2id_path = os.path.join(cfg.log_dir, cfg.exp_name, cfg.preprocess.spk2id)

            with open(spk2id_path, "r") as f:
                self.spk2id = json.load(f)

        if cfg.preprocess.use_text or cfg.preprocess.use_phone:
            self.utt2seq = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                if cfg.preprocess.use_text:
                    text = utt_info["Text"]
                    sequence = text_to_sequence(text, cfg.preprocess.text_cleaners)
                if cfg.preprocess.use_phone:
                    phone = utt_info["Phone"]
                    sequence = text_to_sequence(phone, cfg.preprocess.text_cleaners)

                self.utt2seq[utt] = sequence

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()
        if self.cfg.preprocess.use_spkid:
            single_feature["spk_id"] = np.array(
                [self.spk2id[self.utt2spk[utt]]], dtype=np.int32
            )

        if self.cfg.preprocess.use_text or self.cfg.preprocess.use_phone:
            single_feature["text_seq"] = np.array(self.utt2seq[utt])
            single_feature["text_len"] = len(self.utt2seq[utt])

        return single_feature

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata

    def __len__(self):
        return len(self.metadata)


class VITSTestCollator(BaseTestCollator):
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
            elif key == "text_len":
                packed_batch_features["text_len"] = torch.LongTensor(
                    [b["text_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["text_len"], 1), dtype=torch.long) for b in batch
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
