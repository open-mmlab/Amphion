# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import torchaudio
import numpy as np
import torch
from utils.data_utils import *
from torch.nn.utils.rnn import pad_sequence
from text import text_to_sequence
from text.text_token_collation import phoneIDCollation
from processors.acoustic_extractor import cal_normalized_mel

from models.base.base_dataset import (
    BaseOfflineDataset,
    BaseOfflineCollator,
    BaseTestDataset,
    BaseTestCollator,
)

from processors.content_extractor import (
    ContentvecExtractor,
    WenetExtractor,
    WhisperExtractor,
)


class TTSDataset(BaseOfflineDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """

        assert isinstance(dataset, str)

        self.cfg = cfg

        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)
        self.metadata = self.get_metadata()

        """
        load spk2id and utt2spk from json file
            spk2id: {spk1: 0, spk2: 1, ...}
            utt2spk: {dataset_uid: spk1, ...}
        """
        if cfg.preprocess.use_spkid:
            dataset = self.metadata[0]["Dataset"]

            spk2id_path = os.path.join(processed_data_dir, cfg.preprocess.spk2id)
            with open(spk2id_path, "r") as f:
                self.spk2id = json.load(f)

            utt2spk_path = os.path.join(processed_data_dir, cfg.preprocess.utt2spk)
            self.utt2spk = dict()
            with open(utt2spk_path, "r") as f:
                for line in f.readlines():
                    utt, spk = line.strip().split("\t")
                    self.utt2spk[utt] = spk

        if cfg.preprocess.use_uv:
            self.utt2uv_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                self.utt2uv_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.uv_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_frame_pitch:
            self.utt2frame_pitch_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2frame_pitch_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.pitch_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_frame_energy:
            self.utt2frame_energy_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2frame_energy_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.energy_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_mel:
            self.utt2mel_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2mel_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.mel_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_linear:
            self.utt2linear_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2linear_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.linear_dir,
                    uid + ".npy",
                )

        if cfg.preprocess.use_audio:
            self.utt2audio_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                if cfg.preprocess.extract_audio:
                    self.utt2audio_path[utt] = os.path.join(
                        cfg.preprocess.processed_dir,
                        dataset,
                        cfg.preprocess.audio_dir,
                        uid + ".wav",
                    )
                else:
                    self.utt2audio_path[utt] = utt_info["Path"]

                    # self.utt2audio_path[utt] = os.path.join(
                    #     cfg.preprocess.processed_dir,
                    #     dataset,
                    #     cfg.preprocess.audio_dir,
                    #     uid + ".numpy",
                    # )

        elif cfg.preprocess.use_label:
            self.utt2label_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2label_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.label_dir,
                    uid + ".npy",
                )
        elif cfg.preprocess.use_one_hot:
            self.utt2one_hot_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2one_hot_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.one_hot_dir,
                    uid + ".npy",
                )

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

                    if cfg.preprocess.add_blank:
                        sequence = intersperse(sequence, 0)

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

        if self.cfg.preprocess.use_mel:
            mel = np.load(self.utt2mel_path[utt])
            assert mel.shape[0] == self.cfg.preprocess.n_mel  # [n_mels, T]
            if self.cfg.preprocess.use_min_max_norm_mel:
                # do mel norm
                mel = cal_normalized_mel(mel, utt_info["Dataset"], self.cfg.preprocess)

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]
            single_feature["mel"] = mel.T  # [T, n_mels]

        if self.cfg.preprocess.use_linear:
            linear = np.load(self.utt2linear_path[utt])
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = linear.shape[1]
            single_feature["linear"] = linear.T  # [T, n_linear]

        if self.cfg.preprocess.use_frame_pitch:
            frame_pitch_path = self.utt2frame_pitch_path[utt]
            frame_pitch = np.load(frame_pitch_path)
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_pitch)
            aligned_frame_pitch = align_length(
                frame_pitch, single_feature["target_len"]
            )
            single_feature["frame_pitch"] = aligned_frame_pitch

            if self.cfg.preprocess.use_uv:
                frame_uv_path = self.utt2uv_path[utt]
                frame_uv = np.load(frame_uv_path)
                aligned_frame_uv = align_length(frame_uv, single_feature["target_len"])
                aligned_frame_uv = [
                    0 if frame_uv else 1 for frame_uv in aligned_frame_uv
                ]
                aligned_frame_uv = np.array(aligned_frame_uv)
                single_feature["frame_uv"] = aligned_frame_uv

        if self.cfg.preprocess.use_frame_energy:
            frame_energy_path = self.utt2frame_energy_path[utt]
            frame_energy = np.load(frame_energy_path)
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_energy)
            aligned_frame_energy = align_length(
                frame_energy, single_feature["target_len"]
            )
            single_feature["frame_energy"] = aligned_frame_energy

        if self.cfg.preprocess.use_audio:
            audio, sr = torchaudio.load(self.utt2audio_path[utt])
            audio = audio.cpu().numpy().squeeze()
            single_feature["audio"] = audio
            single_feature["audio_len"] = audio.shape[0]

        if self.cfg.preprocess.use_phone or self.cfg.preprocess.use_text:
            single_feature["phone_seq"] = np.array(self.utt2seq[utt])
            single_feature["phone_len"] = len(self.utt2seq[utt])

        return single_feature

    def __len__(self):
        return super().__len__()

    def get_metadata(self):
        return super().get_metadata()


class TTSCollator(BaseOfflineCollator):
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
