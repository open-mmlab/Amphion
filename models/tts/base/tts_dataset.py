# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from models.base.new_dataset import BaseDataset, BaseTestDataset
from processors.acoustic_extractor import cal_normalized_mel, load_mel_extrema
from processors.content_extractor import (
    ContentvecExtractor,
    WenetExtractor,
    WhisperExtractor,
)
from utils.data_utils import (
    align_content_feature_length,
    align_length,
    load_content_feature_path,
    pitch_shift_to_target,
    transpose_key,
)

EPS = 1.0e-12


class TTSDataset(BaseDataset):
    def __init__(self, args, cfg, is_valid=False):
        super().__init__(args, cfg, is_valid)
        pass


class TTSCollator:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        return packed_batch_features


class TTSTestDataset(BaseTestDataset):
    def __init__(self, args, cfg, infer_type):
        BaseTestDataset.__init__(self, args, cfg, infer_type)
        self.metadata = self.get_metadata()

        target_singer = args.target_singers
        self.cfg = cfg
        self.trans_key = args.trans_key
        assert type(target_singer) == str

        self.target_singer = target_singer.split("_")[-1]
        self.target_dataset = target_singer.replace(
            "_{}".format(self.target_singer), ""
        )

        self.target_mel_extrema = load_mel_extrema(cfg.preprocess, self.target_dataset)
        self.target_mel_extrema = torch.as_tensor(
            self.target_mel_extrema[0]
        ), torch.as_tensor(self.target_mel_extrema[1])

        ######### Load source acoustic features #########
        if cfg.preprocess.use_spkid:
            spk2id_path = os.path.join(args.acoustics_dir, cfg.preprocess.spk2id)
            # utt2sp_path = os.path.join(self.data_root, cfg.preprocess.utt2spk)

            with open(spk2id_path, "r") as f:
                self.spk2id = json.load(f)
            # print("self.spk2id", self.spk2id)

        if cfg.preprocess.use_uv:
            self.utt2uv_path = {
                f'{utt_info["Dataset"]}_{utt_info["Uid"]}': os.path.join(
                    cfg.preprocess.processed_dir,
                    utt_info["Dataset"],
                    cfg.preprocess.uv_dir,
                    utt_info["Uid"] + ".npy",
                )
                for utt_info in self.metadata
            }

        if cfg.preprocess.use_frame_pitch:
            self.utt2frame_pitch_path = {
                f'{utt_info["Dataset"]}_{utt_info["Uid"]}': os.path.join(
                    cfg.preprocess.processed_dir,
                    utt_info["Dataset"],
                    cfg.preprocess.pitch_dir,
                    utt_info["Uid"] + ".npy",
                )
                for utt_info in self.metadata
            }

        if cfg.preprocess.use_frame_energy:
            self.utt2frame_energy_path = {
                f'{utt_info["Dataset"]}_{utt_info["Uid"]}': os.path.join(
                    cfg.preprocess.processed_dir,
                    utt_info["Dataset"],
                    cfg.preprocess.energy_dir,
                    utt_info["Uid"] + ".npy",
                )
                for utt_info in self.metadata
            }

        if cfg.preprocess.use_mel:
            self.utt2mel_path = {
                f'{utt_info["Dataset"]}_{utt_info["Uid"]}': os.path.join(
                    cfg.preprocess.processed_dir,
                    utt_info["Dataset"],
                    cfg.preprocess.mel_dir,
                    utt_info["Uid"] + ".npy",
                )
                for utt_info in self.metadata
            }

        statistics_path = os.path.join(
            cfg.preprocess.processed_dir,
            self.target_dataset,
            cfg.preprocess.pitch_dir,
            "statistics.json",
        )

        self.target_pitch_median = json.load(open(statistics_path, "r"))[
            f"{self.target_dataset}_{self.target_singer}"
        ]["voiced_positions"]["median"]

        ######### Load source content features' path #########
        if cfg.model.condition_encoder.use_whisper:
            self.whisper_aligner = WhisperExtractor(cfg)
            self.utt2whisper_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.whisper_dir
            )

        if cfg.model.condition_encoder.use_contentvec:
            self.contentvec_aligner = ContentvecExtractor(cfg)
            self.utt2contentVec_path = load_content_feature_path(
                self.metadata,
                cfg.preprocess.processed_dir,
                cfg.preprocess.contentvec_dir,
            )

        if cfg.model.condition_encoder.use_mert:
            self.utt2mert_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.mert_dir
            )
        if cfg.model.condition_encoder.use_wenet:
            self.wenet_aligner = WenetExtractor(cfg)
            self.utt2wenet_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.wenet_dir
            )

    def __getitem__(self, index):
        single_feature = {}

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        source_dataset = self.metadata[index]["Dataset"]

        if self.cfg.preprocess.use_spkid:
            single_feature["spk_id"] = np.array(
                [self.spk2id[f"{self.target_dataset}_{self.target_singer}"]],
                dtype=np.int32,
            )

        ######### Get Acoustic Features Item #########
        if self.cfg.preprocess.use_mel:
            mel = np.load(self.utt2mel_path[utt])
            assert mel.shape[0] == self.cfg.preprocess.n_mel  # [n_mels, T]
            if self.cfg.preprocess.use_min_max_norm_mel:
                # mel norm
                mel = cal_normalized_mel(mel, source_dataset, self.cfg.preprocess)

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]
            single_feature["mel"] = mel.T  # [T, n_mels]

        if self.cfg.preprocess.use_frame_pitch:
            frame_pitch_path = self.utt2frame_pitch_path[utt]
            frame_pitch = np.load(frame_pitch_path)

            if self.trans_key:
                try:
                    self.trans_key = int(self.trans_key)
                except:
                    pass
                if type(self.trans_key) == int:
                    frame_pitch = transpose_key(frame_pitch, self.trans_key)
                elif self.trans_key:
                    assert self.target_singer

                    frame_pitch = pitch_shift_to_target(
                        frame_pitch, self.target_pitch_median
                    )

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

        ######### Get Content Features Item #########
        if self.cfg.model.condition_encoder.use_whisper:
            assert "target_len" in single_feature.keys()
            aligned_whisper_feat = self.whisper_aligner.offline_align(
                np.load(self.utt2whisper_path[utt]), single_feature["target_len"]
            )
            single_feature["whisper_feat"] = aligned_whisper_feat

        if self.cfg.model.condition_encoder.use_contentvec:
            assert "target_len" in single_feature.keys()
            aligned_contentvec = self.contentvec_aligner.offline_align(
                np.load(self.utt2contentVec_path[utt]), single_feature["target_len"]
            )
            single_feature["contentvec_feat"] = aligned_contentvec

        if self.cfg.model.condition_encoder.use_mert:
            assert "target_len" in single_feature.keys()
            aligned_mert_feat = align_content_feature_length(
                np.load(self.utt2mert_path[utt]),
                single_feature["target_len"],
                source_hop=self.cfg.preprocess.mert_hop_size,
            )
            single_feature["mert_feat"] = aligned_mert_feat

        if self.cfg.model.condition_encoder.use_wenet:
            assert "target_len" in single_feature.keys()
            aligned_wenet_feat = self.wenet_aligner.offline_align(
                np.load(self.utt2wenet_path[utt]), single_feature["target_len"]
            )
            single_feature["wenet_feat"] = aligned_wenet_feat

        return single_feature

    def __len__(self):
        return len(self.metadata)


class TTSTestCollator:
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
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
