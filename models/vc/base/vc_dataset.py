# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
import json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import resemblyzer
from utils.data_utils import *
from processors.acoustic_extractor import cal_normalized_mel, load_mel_extrema
from processors.content_extractor import (
    ContentvecExtractor,
    WhisperExtractor,
    WenetExtractor,
    HubertExtractor,
)
from models.base.base_dataset import (
    BaseCollator,
    BaseDataset,
)
from models.base.new_dataset import BaseTestDataset

EPS = 1.0e-12


class VCDataset(BaseDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        BaseDataset.__init__(self, cfg, dataset, is_valid=is_valid)

        cfg = self.cfg
        if cfg.preprocess.segment_size is not None:
            metadata_new = []
            for item in self.metadata:
                if (
                    item["Duration"] * cfg.preprocess.sample_rate
                    > cfg.preprocess.segment_size
                ):
                    metadata_new.append(item)
            self.metadata = metadata_new

        if cfg.model.condition_encoder.use_whisper:
            self.whisper_aligner = WhisperExtractor(self.cfg)
            self.utt2whisper_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.whisper_dir
            )

        if cfg.model.condition_encoder.use_contentvec:
            self.contentvec_aligner = ContentvecExtractor(self.cfg)
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
            self.wenet_aligner = WenetExtractor(self.cfg)
            self.utt2wenet_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.wenet_dir
            )
        if cfg.model.condition_encoder.use_hubert:
            self.hubert_aligner = HubertExtractor(self.cfg)
            self.utt2hubert_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.hubert_dir
            )

    def __getitem__(self, index):
        single_feature = BaseDataset.__getitem__(self, index)

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        if self.cfg.preprocess.use_frame_pitch:
            assert "frame_pitch" in single_feature.keys()
            scaler = MinMaxScaler()
            scaler.fit(single_feature["frame_pitch"].reshape(-1, 1))
            single_feature["frame_pitch"] = scaler.transform(
                single_feature["frame_pitch"].reshape(-1, 1)
            )
            single_feature["frame_pitch"] = single_feature["frame_pitch"].reshape(-1)

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

        if self.cfg.model.condition_encoder.use_hubert:
            assert "target_len" in single_feature.keys()
            aligned_hubert_feat = self.hubert_aligner.offline_align(
                np.load(self.utt2hubert_path[utt]), single_feature["target_len"]
            )
            single_feature["hubert_feat"] = aligned_hubert_feat.astype(np.int32)

        # print(single_feature.keys())
        # for k, v in single_feature.items():
        #     if type(v) in [torch.Tensor, np.ndarray]:
        #         print(k, v.shape)
        #     else:
        #         print(k, v)
        # exit()

        return self.clip_if_too_long(single_feature)

    def __len__(self):
        return len(self.metadata)

    def random_select(self, feature_seq_len, max_seq_len, ending_ts=2812):
        """
        ending_ts: to avoid invalid whisper features for over 30s audios
            2812 = 30 * 24000 // 256
        """
        ts = max(feature_seq_len - max_seq_len, 0)
        ts = min(ts, ending_ts - max_seq_len)

        start = random.randint(0, ts)
        end = start + max_seq_len
        return start, end

    def clip_if_too_long(self, sample, max_seq_len=512):
        """
        sample :
            {
                'spk_id': (1,),
                'target_len': int
                'mel': (seq_len, dim),
                'frame_pitch': (seq_len,)
                'frame_energy': (seq_len,)
                'content_vector_feat': (seq_len, dim)
            }
        """

        if sample["target_len"] <= max_seq_len:
            return sample

        start, end = self.random_select(sample["target_len"], max_seq_len)
        sample["target_len"] = end - start

        for k in sample.keys():
            if k == "audio":
                # audio should be clipped in hop_size scale
                sample[k] = sample[k][
                    start
                    * self.cfg.preprocess.hop_size : end
                    * self.cfg.preprocess.hop_size
                ]
            elif k == "audio_len":
                sample[k] = (end - start) * self.cfg.preprocess.hop_size
            elif k not in ["spk_id", "target_len", "spkemb"]:
                sample[k] = sample[k][start:end]

        return sample


class VCCollator(BaseCollator):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)

    def __call__(self, batch):
        parsed_batch_features = BaseCollator.__call__(self, batch)
        return parsed_batch_features


class VCTestDataset(BaseTestDataset):
    def __init__(self, args, cfg, infer_type):
        BaseTestDataset.__init__(self, args, cfg, infer_type)
        self.metadata = self.get_metadata()

        self.target = args.target
        self.cfg = cfg
        self.trans_key = args.trans_key

        self.target_dataset = cfg.dataset[0]
        if cfg.preprocess.mel_min_max_norm:
            self.target_mel_extrema = load_mel_extrema(
                cfg.preprocess, self.target_dataset
            )
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

        if cfg.preprocess.use_spkemb:
            self.utt2spk_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2spk_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.speaker_dir,
                    uid + ".npy",
                )

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
        if cfg.model.condition_encoder.use_hubert:
            self.hubert_aligner = HubertExtractor(cfg)
            self.utt2hubert_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.hubert_dir
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

        if self.cfg.preprocess.use_spkemb:
            voice_encoder = resemblyzer.VoiceEncoder("cpu", verbose=False)
            target_wav = resemblyzer.preprocess_wav(self.target)
            single_feature["spkemb"] = voice_encoder.embed_utterance(target_wav)

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

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_pitch)
            aligned_frame_pitch = align_length(
                frame_pitch, single_feature["target_len"]
            )
            single_feature["frame_pitch"] = aligned_frame_pitch
            scaler = MinMaxScaler()
            scaler.fit(single_feature["frame_pitch"].reshape(-1, 1))
            single_feature["frame_pitch"] = scaler.transform(
                single_feature["frame_pitch"].reshape(-1, 1)
            )
            single_feature["frame_pitch"] = single_feature["frame_pitch"].reshape(-1)

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

        if self.cfg.model.condition_encoder.use_hubert:
            assert "target_len" in single_feature.keys()
            aligned_hubert_feat = self.hubert_aligner.offline_align(
                np.load(self.utt2hubert_path[utt]), single_feature["target_len"]
            )
            single_feature["hubert_feat"] = aligned_hubert_feat.astype(np.int32)

        return single_feature

    def __len__(self):
        return len(self.metadata)


class VCTestCollator:
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
