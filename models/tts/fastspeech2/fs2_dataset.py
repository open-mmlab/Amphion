# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.base.base_dataset import (
    BaseOfflineCollator,
    BaseOfflineDataset,
    BaseTestDataset,
    BaseTestCollator,
)
from text import text_to_sequence


class FS2Dataset(BaseOfflineDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        BaseOfflineDataset.__init__(self, cfg, dataset, is_valid=is_valid)
        self.batch_size = cfg.train.batch_size
        cfg = cfg.preprocess
        # utt2duration
        self.utt2duration_path = {}
        for utt_info in self.metadata:
            dataset = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset, uid)

            self.utt2duration_path[utt] = os.path.join(
                cfg.processed_dir,
                dataset,
                cfg.duration_dir,
                uid + ".npy",
            )
        self.utt2dur = self.read_duration()

        if cfg.use_frame_energy:
            self.frame_utt2energy, self.energy_statistic = load_energy(
                self.metadata,
                cfg.processed_dir,
                cfg.energy_dir,
                use_log_scale=cfg.use_log_scale_energy,
                utt2spk=self.preprocess.utt2spk if cfg.use_spkid else None,
                return_norm=True,
            )
        elif cfg.use_phone_energy:
            self.phone_utt2energy, self.energy_statistic = load_energy(
                self.metadata,
                cfg.processed_dir,
                cfg.phone_energy_dir,
                use_log_scale=cfg.use_log_scale_energy,
                utt2spk=self.utt2spk if cfg.use_spkid else None,
                return_norm=True,
            )

        if cfg.use_frame_pitch:
            self.frame_utt2pitch, self.pitch_statistic = load_energy(
                self.metadata,
                cfg.processed_dir,
                cfg.pitch_dir,
                use_log_scale=cfg.energy_extract_mode,
                utt2spk=self.utt2spk if cfg.use_spkid else None,
                return_norm=True,
            )

        elif cfg.use_phone_pitch:
            self.phone_utt2pitch, self.pitch_statistic = load_energy(
                self.metadata,
                cfg.processed_dir,
                cfg.phone_pitch_dir,
                use_log_scale=cfg.use_log_scale_pitch,
                utt2spk=self.utt2spk if cfg.use_spkid else None,
                return_norm=True,
            )

        # utt2lab
        self.utt2lab_path = {}
        for utt_info in self.metadata:
            dataset = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset, uid)

            self.utt2lab_path[utt] = os.path.join(
                cfg.processed_dir,
                dataset,
                cfg.lab_dir,
                uid + ".txt",
            )

        self.speaker_map = {}
        if os.path.exists(os.path.join(cfg.processed_dir, "spk2id.json")):
            with open(
                os.path.exists(os.path.join(cfg.processed_dir, "spk2id.json"))
            ) as f:
                self.speaker_map = json.load(f)

        self.metadata = self.check_metadata()

    def __getitem__(self, index):
        single_feature = BaseOfflineDataset.__getitem__(self, index)

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        duration = self.utt2dur[utt]

        # text
        f = open(self.utt2lab_path[utt], "r")
        phones = f.readlines()[0].strip()
        f.close()
        # todo: add cleaner(chenxi)
        phones_ids = np.array(text_to_sequence(phones, ["english_cleaners"]))
        text_len = len(phones_ids)

        if self.cfg.preprocess.use_frame_pitch:
            pitch = self.frame_utt2pitch[utt]
        elif self.cfg.preprocess.use_phone_pitch:
            pitch = self.phone_utt2pitch[utt]

        if self.cfg.preprocess.use_frame_energy:
            energy = self.frame_utt2energy[utt]
        elif self.cfg.preprocess.use_phone_energy:
            energy = self.phone_utt2energy[utt]

        # speaker
        if len(self.speaker_map) > 0:
            speaker_id = self.speaker_map[utt_info["Singer"]]
        else:
            speaker_id = 0

        single_feature.update(
            {
                "durations": duration,
                "texts": phones_ids,
                "spk_id": speaker_id,
                "text_len": text_len,
                "pitch": pitch,
                "energy": energy,
                "uid": uid,
            }
        )
        return self.clip_if_too_long(single_feature)

    def read_duration(self):
        # read duration
        utt2dur = {}
        for index in range(len(self.metadata)):
            utt_info = self.metadata[index]
            dataset = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset, uid)

            if not os.path.exists(self.utt2mel_path[utt]) or not os.path.exists(
                self.utt2duration_path[utt]
            ):
                continue

            mel = np.load(self.utt2mel_path[utt]).transpose(1, 0)
            duration = np.load(self.utt2duration_path[utt])
            assert mel.shape[0] == sum(
                duration
            ), f"{utt}: mismatch length between mel {mel.shape[0]} and sum(duration) {sum(duration)}"
            utt2dur[utt] = duration
        return utt2dur

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

    def clip_if_too_long(self, sample, max_seq_len=1000):
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
            if k not in ["spk_id", "target_len"]:
                sample[k] = sample[k][start:end]

        return sample

    def check_metadata(self):
        new_metadata = []
        for utt_info in self.metadata:
            dataset = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset, uid)
            if not os.path.exists(self.utt2duration_path[utt]) or not os.path.exists(
                self.utt2mel_path[utt]
            ):
                continue
            else:
                new_metadata.append(utt_info)
        return new_metadata


class FS2Collator(BaseOfflineCollator):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        BaseOfflineCollator.__init__(self, cfg)
        self.sort = cfg.train.sort_sample
        self.batch_size = cfg.train.batch_size
        self.drop_last = cfg.train.drop_last

    def __call__(self, batch):
        # mel: [b, T, n_mels]
        # frame_pitch, frame_energy: [1, T]
        # target_len: [1]
        # spk_id: [b, 1]
        # mask: [b, T, 1]
        packed_batch_features = dict()

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
                packed_batch_features["text_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "spk_id":
                packed_batch_features["spk_id"] = torch.LongTensor(
                    [b["spk_id"] for b in batch]
                )
            elif key == "uid":
                packed_batch_features[key] = [b["uid"] for b in batch]
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )
        return packed_batch_features


class FS2TestDataset(BaseTestDataset):
    def __init__(self, args, cfg, infer_type=None):
        datasets = cfg.dataset
        cfg = cfg.preprocess
        is_bigdata = False

        assert len(datasets) >= 1
        if len(datasets) > 1:
            datasets.sort()
            bigdata_version = "_".join(datasets)
            processed_data_dir = os.path.join(cfg.processed_dir, bigdata_version)
            is_bigdata = True
        else:
            processed_data_dir = os.path.join(cfg.processed_dir, args.dataset)

        if args.test_list_file:
            self.metafile_path = args.test_list_file
            self.metadata = self.get_metadata()
        else:
            assert args.testing_set
            source_metafile_path = os.path.join(
                cfg.processed_dir,
                args.dataset,
                "{}.json".format(args.testing_set),
            )
            with open(source_metafile_path, "r") as f:
                self.metadata = json.load(f)

        self.cfg = cfg
        self.datasets = datasets
        self.data_root = processed_data_dir
        self.is_bigdata = is_bigdata
        self.source_dataset = args.dataset

        ######### Load source acoustic features #########
        if cfg.use_spkid:
            spk2id_path = os.path.join(self.data_root, cfg.spk2id)
            utt2sp_path = os.path.join(self.data_root, cfg.utt2spk)
            self.spk2id, self.utt2spk = get_spk_map(spk2id_path, utt2sp_path, datasets)

        # utt2lab
        self.utt2lab_path = {}
        for utt_info in self.metadata:
            dataset = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset, uid)
            self.utt2lab_path[utt] = os.path.join(
                cfg.processed_dir,
                dataset,
                cfg.lab_dir,
                uid + ".txt",
            )

        self.speaker_map = {}
        if os.path.exists(os.path.join(cfg.processed_dir, "spk2id.json")):
            with open(
                os.path.exists(os.path.join(cfg.processed_dir, "spk2id.json"))
            ) as f:
                self.speaker_map = json.load(f)

    def __getitem__(self, index):
        single_feature = {}

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        # text
        f = open(self.utt2lab_path[utt], "r")
        phones = f.readlines()[0].strip()
        f.close()

        phones_ids = np.array(text_to_sequence(phones, self.cfg.text_cleaners))
        text_len = len(phones_ids)

        # speaker
        if len(self.speaker_map) > 0:
            speaker_id = self.speaker_map[utt_info["Singer"]]
        else:
            speaker_id = 0

        single_feature.update(
            {
                "texts": phones_ids,
                "spk_id": speaker_id,
                "text_len": text_len,
            }
        )

        return single_feature

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata


class FS2TestCollator(BaseTestCollator):
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
                packed_batch_features["text_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "spk_id":
                packed_batch_features["spk_id"] = torch.LongTensor(
                    [b["spk_id"] for b in batch]
                )
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
