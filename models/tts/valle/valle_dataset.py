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
                    cfg.preprocess.acoustic_token_dir,   # code
                    uid + ".npy",
                )

    
    def __len__(self):
        return super().__len__()

    def get_metadata(self):
        metadata_filter = []
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for utt_info in metadata:
            duration = utt_info['Duration']
            if duration >= self.cfg.preprocess.max_duration or duration <= self.cfg.preprocess.min_duration:
                continue
            metadata_filter.append(utt_info)

        return metadata_filter

    def get_dur(self, idx):
        utt_info = self.metadata[idx]
        return utt_info['Duration']


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

class VALLECollator(TTSCollator):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def __call__(self, batch):
        parsed_batch_features = super().__call__(batch)
        return parsed_batch_features

class VALLETestDataset(TTSTestDataset):
    def __init__(self,args, cfg):
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
                audio_prompt_token = encoded_frames[0][0].transpose(2, 1).squeeze(0).cpu().numpy()
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
