# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.tts.base.tts_dataset import (
    TTSDataset,
    TTSCollator,
    TTSTestDataset,
    TTSTestCollator,
)

from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)

from utils.tokenizer import tokenize_audio


class VALLEDataset(TTSDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        super().__init__(cfg, dataset, is_valid=is_valid)

        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """

        assert isinstance(dataset, str)

        assert cfg.preprocess.use_acoustic_token == True
        if cfg.preprocess.use_acoustic_token:
            self.utt2acousticToken_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2acousticToken_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.acoustic_token_dir,  # code
                    uid + ".npy",
                )

        self.all_num_frames = []
        for i in range(len(self.metadata)):
            self.all_num_frames.append(self.metadata[i]["Duration"])
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )

    def __len__(self):
        return super().__len__()

    def get_metadata(self):
        metadata_filter = []
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for utt_info in metadata:
            duration = utt_info["Duration"]
            if (
                duration >= self.cfg.preprocess.max_duration
                or duration <= self.cfg.preprocess.min_duration
            ):
                continue
            metadata_filter.append(utt_info)

        return metadata_filter

    def get_dur(self, idx):
        utt_info = self.metadata[idx]
        return utt_info["Duration"]

    def __getitem__(self, index):
        single_feature = super().__getitem__(index)

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        # acoustic token
        if self.cfg.preprocess.use_acoustic_token:
            acoustic_token = np.load(self.utt2acousticToken_path[utt])
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = acoustic_token.shape[0]
            single_feature["acoustic_token"] = acoustic_token  # [T, 8]

        return single_feature

    def get_num_frames(self, index):
        utt_info = self.metadata[index]
        return int(
            utt_info["Duration"]
            * (self.cfg.preprocess.sample_rate // self.cfg.preprocess.codec_hop_size)
        )


class VALLECollator(TTSCollator):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, batch):
        parsed_batch_features = super().__call__(batch)
        return parsed_batch_features


class VALLETestDataset(TTSTestDataset):
    def __init__(self, args, cfg):
        super().__init__(args, cfg)

        # prepare data
        assert cfg.preprocess.use_acoustic_token == True
        if cfg.preprocess.use_acoustic_token:
            self.utt2acousticToken = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                # extract acoustic token
                audio_file = utt_info["Audio_pormpt_path"]
                encoded_frames = tokenize_audio(self.audio_tokenizer, audio_file)
                audio_prompt_token = (
                    encoded_frames[0][0].transpose(2, 1).squeeze(0).cpu().numpy()
                )
                self.utt2acousticToken[utt] = audio_prompt_token

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        # acoustic token
        if self.cfg.preprocess.use_acoustic_token:
            acoustic_token = self.utt2acousticToken[utt]
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = acoustic_token.shape[0]
            single_feature["acoustic_token"] = acoustic_token  # [T, 8]

        # phone sequence todo
        if self.cfg.preprocess.use_phone:
            single_feature["phone_seq"] = np.array(self.utt2seq[utt])
            single_feature["phone_len"] = len(self.utt2seq[utt])
            single_feature["pmt_phone_seq"] = np.array(self.utt2pmtseq[utt])
            single_feature["pmt_phone_len"] = len(self.utt2pmtseq[utt])

        return single_feature

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata

    def __len__(self):
        return len(self.metadata)


class VALLETestCollator(TTSTestCollator):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
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
            elif key == "pmt_phone_len":
                packed_batch_features["pmt_phone_len"] = torch.LongTensor(
                    [b["pmt_phone_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["pmt_phone_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["pmt_phone_len_mask"] = pad_sequence(
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
