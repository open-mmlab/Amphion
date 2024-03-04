# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from processors.acoustic_extractor import cal_normalized_mel
from processors.acoustic_extractor import load_normalized
from models.base.base_dataset import (
    BaseOfflineCollator,
    BaseOfflineDataset,
    BaseTestDataset,
    BaseTestCollator,
)
from text import text_to_sequence
from text.cmudict import valid_symbols
from tqdm import tqdm
import pickle


class NS2Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        assert isinstance(dataset, str)

        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)

        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        # train.json

        self.metafile_path = os.path.join(processed_data_dir, meta_file)

        self.metadata = self.get_metadata()

        self.cfg = cfg

        assert cfg.preprocess.use_mel == False
        if cfg.preprocess.use_mel:
            self.utt2melspec_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2melspec_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.melspec_dir,  # mel
                    utt_info["speaker"],
                    uid + ".npy",
                )

        assert cfg.preprocess.use_code == True
        if cfg.preprocess.use_code:
            self.utt2code_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2code_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.code_dir,  # code
                    utt_info["speaker"],
                    uid + ".npy",
                )

        assert cfg.preprocess.use_spkid == True
        if cfg.preprocess.use_spkid:
            self.utt2spkid = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2spkid[utt] = utt_info["speaker"]

        assert cfg.preprocess.use_pitch == True
        if cfg.preprocess.use_pitch:
            self.utt2pitch_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2pitch_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.pitch_dir,  # pitch
                    utt_info["speaker"],
                    uid + ".npy",
                )

        assert cfg.preprocess.use_duration == True
        if cfg.preprocess.use_duration:
            self.utt2duration_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2duration_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.duration_dir,  # duration
                    utt_info["speaker"],
                    uid + ".npy",
                )

        assert cfg.preprocess.use_phone == True
        if cfg.preprocess.use_phone:
            self.utt2phone = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2phone[utt] = utt_info["phones"]

        assert cfg.preprocess.use_len == True
        if cfg.preprocess.use_len:
            self.utt2len = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2len[utt] = utt_info["num_frames"]

        # for cross reference
        if cfg.preprocess.use_cross_reference:
            self.spkid2utt = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                spkid = utt_info["speaker"]
                if spkid not in self.spkid2utt:
                    self.spkid2utt[spkid] = []
                self.spkid2utt[spkid].append(utt)

        # get phone to id / id to phone map
        self.phone2id, self.id2phone = self.get_phone_map()

        self.all_num_frames = []
        for i in range(len(self.metadata)):
            self.all_num_frames.append(self.metadata[i]["num_frames"])
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )

    def __len__(self):
        return len(self.metadata)

    def get_dataset_name(self):
        return self.metadata[0]["Dataset"]

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("metadata len: ", len(metadata))

        return metadata

    def get_phone_map(self):
        symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
        phone2id = {s: i for i, s in enumerate(symbols)}
        id2phone = {i: s for s, i in phone2id.items()}
        return phone2id, id2phone

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        if self.cfg.preprocess.read_metadata:
            metadata_uid_path = os.path.join(
                self.cfg.preprocess.processed_dir,
                self.cfg.preprocess.metadata_dir,
                dataset,
                # utt_info["speaker"],
                uid + ".pkl",
            )
            with open(metadata_uid_path, "rb") as f:
                metadata_uid = pickle.load(f)
            # code
            code = metadata_uid["code"]
            # frame_nums
            frame_nums = code.shape[1]
            # pitch
            pitch = metadata_uid["pitch"]
            # duration
            duration = metadata_uid["duration"]
            # phone_id
            phone_id = np.array(
                [
                    *map(
                        self.phone2id.get,
                        self.utt2phone[utt].replace("{", "").replace("}", "").split(),
                    )
                ]
            )

        else:
            # code
            code = np.load(self.utt2code_path[utt])
            # frame_nums
            frame_nums = code.shape[1]
            # pitch
            pitch = np.load(self.utt2pitch_path[utt])
            # duration
            duration = np.load(self.utt2duration_path[utt])
            # phone_id
            phone_id = np.array(
                [
                    *map(
                        self.phone2id.get,
                        self.utt2phone[utt].replace("{", "").replace("}", "").split(),
                    )
                ]
            )

        # align length
        code, pitch, duration, phone_id, frame_nums = self.align_length(
            code, pitch, duration, phone_id, frame_nums
        )

        # spkid
        spkid = self.utt2spkid[utt]

        # get target and reference
        out = self.get_target_and_reference(code, pitch, duration, phone_id, frame_nums)
        code, ref_code = out["code"], out["ref_code"]
        pitch, ref_pitch = out["pitch"], out["ref_pitch"]
        duration, ref_duration = out["duration"], out["ref_duration"]
        phone_id, ref_phone_id = out["phone_id"], out["ref_phone_id"]
        frame_nums, ref_frame_nums = out["frame_nums"], out["ref_frame_nums"]

        # phone_id_frame
        assert len(phone_id) == len(duration)
        phone_id_frame = []
        for i in range(len(phone_id)):
            phone_id_frame.extend([phone_id[i] for _ in range(duration[i])])
        phone_id_frame = np.array(phone_id_frame)

        # ref_phone_id_frame
        assert len(ref_phone_id) == len(ref_duration)
        ref_phone_id_frame = []
        for i in range(len(ref_phone_id)):
            ref_phone_id_frame.extend([ref_phone_id[i] for _ in range(ref_duration[i])])
        ref_phone_id_frame = np.array(ref_phone_id_frame)

        single_feature.update(
            {
                "code": code,
                "frame_nums": frame_nums,
                "pitch": pitch,
                "duration": duration,
                "phone_id": phone_id,
                "phone_id_frame": phone_id_frame,
                "ref_code": ref_code,
                "ref_frame_nums": ref_frame_nums,
                "ref_pitch": ref_pitch,
                "ref_duration": ref_duration,
                "ref_phone_id": ref_phone_id,
                "ref_phone_id_frame": ref_phone_id_frame,
                "spkid": spkid,
            }
        )

        return single_feature

    def get_num_frames(self, index):
        utt_info = self.metadata[index]
        return utt_info["num_frames"]

    def align_length(self, code, pitch, duration, phone_id, frame_nums):
        # aligh lenght of code, pitch, duration, phone_id, and frame nums
        code_len = code.shape[1]
        pitch_len = len(pitch)
        dur_sum = sum(duration)
        min_len = min(code_len, dur_sum)
        code = code[:, :min_len]
        if pitch_len >= min_len:
            pitch = pitch[:min_len]
        else:
            pitch = np.pad(pitch, (0, min_len - pitch_len), mode="edge")
        frame_nums = min_len
        if dur_sum > min_len:
            assert (duration[-1] - (dur_sum - min_len)) >= 0
            duration[-1] = duration[-1] - (dur_sum - min_len)
            assert duration[-1] >= 0

        return code, pitch, duration, phone_id, frame_nums

    def get_target_and_reference(self, code, pitch, duration, phone_id, frame_nums):
        phone_nums = len(phone_id)
        clip_phone_nums = np.random.randint(
            int(phone_nums * 0.1), int(phone_nums * 0.5) + 1
        )
        clip_phone_nums = max(clip_phone_nums, 1)
        assert clip_phone_nums < phone_nums and clip_phone_nums >= 1
        if self.cfg.preprocess.clip_mode == "mid":
            start_idx = np.random.randint(0, phone_nums - clip_phone_nums)
        elif self.cfg.preprocess.clip_mode == "start":
            if duration[0] == 0 and clip_phone_nums == 1:
                start_idx = 1
            else:
                start_idx = 0
        else:
            assert self.cfg.preprocess.clip_mode in ["mid", "start"]
        end_idx = start_idx + clip_phone_nums
        start_frames = sum(duration[:start_idx])
        end_frames = sum(duration[:end_idx])

        new_code = np.concatenate(
            (code[:, :start_frames], code[:, end_frames:]), axis=1
        )
        ref_code = code[:, start_frames:end_frames]

        new_pitch = np.append(pitch[:start_frames], pitch[end_frames:])
        ref_pitch = pitch[start_frames:end_frames]

        new_duration = np.append(duration[:start_idx], duration[end_idx:])
        ref_duration = duration[start_idx:end_idx]

        new_phone_id = np.append(phone_id[:start_idx], phone_id[end_idx:])
        ref_phone_id = phone_id[start_idx:end_idx]

        new_frame_nums = frame_nums - (end_frames - start_frames)
        ref_frame_nums = end_frames - start_frames

        return {
            "code": new_code,
            "ref_code": ref_code,
            "pitch": new_pitch,
            "ref_pitch": ref_pitch,
            "duration": new_duration,
            "ref_duration": ref_duration,
            "phone_id": new_phone_id,
            "ref_phone_id": ref_phone_id,
            "frame_nums": new_frame_nums,
            "ref_frame_nums": ref_frame_nums,
        }


class NS2Collator(BaseOfflineCollator):
    def __init__(self, cfg):
        BaseOfflineCollator.__init__(self, cfg)

    def __call__(self, batch):
        packed_batch_features = dict()

        # code: (B, 16, T)
        # frame_nums: (B,)   not used
        # pitch: (B, T)
        # duration: (B, N)
        # phone_id: (B, N)
        # phone_id_frame: (B, T)
        # ref_code: (B, 16, T')
        # ref_frame_nums: (B,)   not used
        # ref_pitch: (B, T)   not used
        # ref_duration: (B, N')   not used
        # ref_phone_id: (B, N')   not used
        # ref_phone_frame: (B, T')   not used
        # spkid: (B,)   not used
        # phone_mask: (B, N)
        # mask: (B, T)
        # ref_mask: (B, T')

        for key in batch[0].keys():
            if key == "phone_id":
                phone_ids = [torch.LongTensor(b["phone_id"]) for b in batch]
                phone_masks = [torch.ones(len(b["phone_id"])) for b in batch]
                packed_batch_features["phone_id"] = pad_sequence(
                    phone_ids,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["phone_mask"] = pad_sequence(
                    phone_masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "phone_id_frame":
                phone_id_frames = [torch.LongTensor(b["phone_id_frame"]) for b in batch]
                masks = [torch.ones(len(b["phone_id_frame"])) for b in batch]
                packed_batch_features["phone_id_frame"] = pad_sequence(
                    phone_id_frames,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["mask"] = pad_sequence(
                    masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "ref_code":
                ref_codes = [
                    torch.from_numpy(b["ref_code"]).transpose(0, 1) for b in batch
                ]
                ref_masks = [torch.ones(max(b["ref_code"].shape[1], 1)) for b in batch]
                packed_batch_features["ref_code"] = pad_sequence(
                    ref_codes,
                    batch_first=True,
                    padding_value=0,
                ).transpose(1, 2)
                packed_batch_features["ref_mask"] = pad_sequence(
                    ref_masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "code":
                codes = [torch.from_numpy(b["code"]).transpose(0, 1) for b in batch]
                masks = [torch.ones(max(b["code"].shape[1], 1)) for b in batch]
                packed_batch_features["code"] = pad_sequence(
                    codes,
                    batch_first=True,
                    padding_value=0,
                ).transpose(1, 2)
                packed_batch_features["mask"] = pad_sequence(
                    masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "pitch":
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=50.0
                )
            elif key == "duration":
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )
            elif key == "frame_nums":
                packed_batch_features["frame_nums"] = torch.LongTensor(
                    [b["frame_nums"] for b in batch]
                )
            elif key == "ref_frame_nums":
                packed_batch_features["ref_frame_nums"] = torch.LongTensor(
                    [b["ref_frame_nums"] for b in batch]
                )
            else:
                pass

        return packed_batch_features


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
