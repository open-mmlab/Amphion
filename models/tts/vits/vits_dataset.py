# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
from text import text_to_sequence
from text.text_token_collation import phoneIDCollation
from models.tts.base.tts_dataset import (
    TTSDataset,
    TTSCollator,
    TTSTestDataset,
    TTSTestCollator,
)


class VITSDataset(TTSDataset):
    def __init__(self, cfg, dataset, is_valid):
        super().__init__(cfg, dataset, is_valid=is_valid)

    def __getitem__(self, index):
        single_feature = super().__getitem__(index)
        return single_feature

    def __len__(self):
        return super().__len__()

    def get_metadata(self):
        metadata_filter = []
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for utt_info in metadata:
            duration = utt_info["Duration"]
            frame_len = (
                duration
                * self.cfg.preprocess.sample_rate
                // self.cfg.preprocess.hop_size
            )
            if (
                frame_len
                < self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size
            ):
                continue
            metadata_filter.append(utt_info)

        return metadata_filter


class VITSCollator(TTSCollator):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, batch):
        parsed_batch_features = super().__call__(batch)
        return parsed_batch_features


class VITSTestDataset(TTSTestDataset):
    def __init__(self, args, cfg):
        super().__init__(args, cfg)
        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, args.dataset)
        if cfg.preprocess.use_spkid:
            spk2id_path = os.path.join(processed_data_dir, cfg.preprocess.spk2id)
            with open(spk2id_path, "r") as f:
                self.spk2id = json.load(f)

            utt2spk_path = os.path.join(processed_data_dir, cfg.preprocess.utt2spk)
            self.utt2spk = dict()
            with open(utt2spk_path, "r") as f:
                for line in f.readlines():
                    utt, spk = line.strip().split("\t")
                    self.utt2spk[utt] = spk

        if cfg.preprocess.use_text or cfg.preprocess.use_phone:
            self.utt2seq = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                if cfg.preprocess.use_text:
                    text = utt_info["Text"]
                    sequence = text_to_sequence(text, cfg.preprocess.text_cleaners)
                elif cfg.preprocess.use_phone:
                    # load phoneme squence from phone file
                    phone_path = os.path.join(
                        processed_data_dir, cfg.preprocess.phone_dir, uid + ".phone"
                    )
                    with open(phone_path, "r") as fin:
                        phones = fin.readlines()
                        assert len(phones) == 1
                        phones = phones[0].strip()
                    phones_seq = phones.split(" ")

                    phon_id_collator = phoneIDCollation(cfg, dataset=dataset)
                    sequence = phon_id_collator.get_phone_id_sequence(cfg, phones_seq)

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

        if self.cfg.preprocess.use_phone or self.cfg.preprocess.use_text:
            single_feature["phone_seq"] = np.array(self.utt2seq[utt])
            single_feature["phone_len"] = len(self.utt2seq[utt])

        return single_feature

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata

    def __len__(self):
        return len(self.metadata)


class VITSTestCollator(TTSTestCollator):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        return super().__call__(batch)
