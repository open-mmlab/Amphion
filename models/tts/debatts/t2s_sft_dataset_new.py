# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from cmath import inf
import io
import librosa
import torch
import json
import tqdm
import numpy as np
import logging
import pickle
import os

import time
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool
import concurrent.futures
from pathlib import Path
from transformers import SeamlessM4TFeatureExtractor
from transformers import Wav2Vec2BertModel

os.chdir("./models/tts/debatts")
import sys

sys.path.append("./models/tts/debatts")
from utils.g2p_new.g2p_new import new_g2p
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WarningFilter(logging.Filter):
    def filter(self, record):

        if record.name == "phonemizer" and record.levelno == logging.WARNING:
            return False
        if record.name == "qcloud_cos.cos_client" and record.levelno == logging.INFO:
            return False
        if record.name == "jieba" and record.levelno == logging.DEBUG:
            return False
        return True


filter = WarningFilter()
logging.getLogger("phonemizer").addFilter(filter)
logging.getLogger("qcloud_cos.cos_client").addFilter(filter)
logging.getLogger("jieba").addFilter(filter)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T2SDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg=None,
    ):
        self.cfg = cfg

        self.meta_info_path = "Debatts-Data Summary Json"
        with open(self.meta_info_path, "r") as f:
            self.meta_info_data = json.load(f)

        self.wav_paths = []
        self.prompt0_paths = []  # Add prompt0 paths
        self.wav_path_index2duration = []
        self.wav_path_index2phonelen = []
        self.wav_path_index2spkid = []
        self.wav_path_index2phoneid = []
        self.index2num_frames = []
        self.index2lang = []
        self.lang2id = {"en": 1, "zh": 2, "ja": 3, "fr": 4, "ko": 5, "de": 6}

        for info in self.meta_info_data:
            if info["prompt0_wav_path"] == None:
                continue
            self.wav_paths.append(info["wav_path"])
            self.prompt0_paths.append(info["prompt0_wav_path"])  # Add prompt0 path
            self.wav_path_index2duration.append(info["duration"])
            self.wav_path_index2phonelen.append(info["phone_count"])
            self.wav_path_index2spkid.append(info["speaker_id"])
            self.wav_path_index2phoneid.append(info["phone_id"])
            self.index2num_frames.append(info["duration"] * 50 + len(info["phone_id"]))
            lang_id = self.lang2id[info["language"]]
            self.index2lang.append(lang_id)

            # self.index2num_frames.append(info["duration"] * self.cfg.preprocess.sample_rate)

        self.num_frame_indices = np.array(
            sorted(
                range(len(self.index2num_frames)),
                key=lambda k: self.index2num_frames[k],
            )
        )

        self.processor = SeamlessM4TFeatureExtractor.from_pretrained("./w2v-bert-2")

    def new_g2p(self, text, language):
        return new_g2p(text, language)

    def __len__(self):
        return self.wav_paths.__len__()

    def get_num_frames(self, index):
        return (
            self.wav_path_index2duration[index] * 50
            + self.wav_path_index2phonelen[index]
        )

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        speech, sr = librosa.load(wav_path, sr=self.cfg.preprocess.sample_rate)
        speech = np.pad(
            speech,
            (
                0,
                self.cfg.preprocess.hop_size
                - len(speech) % self.cfg.preprocess.hop_size,
            ),
            mode="constant",
        )
        # resample the speech to 16k for feature extraction
        if self.cfg.preprocess.sample_rate != 16000:
            speech_16k = librosa.resample(
                speech, orig_sr=self.cfg.preprocess.sample_rate, target_sr=16000
            )
        else:
            speech_16k = speech
        inputs = self.processor(speech_16k, sampling_rate=16000)
        # wav 2 bert convert to useful feature
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]

        prompt0_wav_path = self.prompt0_paths[idx]  # Get prompt0 path
        speech_prompt0, sr_prompt0 = librosa.load(
            prompt0_wav_path, sr=self.cfg.preprocess.sample_rate
        )
        speech_prompt0 = np.pad(
            speech_prompt0,
            (
                0,
                self.cfg.preprocess.hop_size
                - len(speech_prompt0) % self.cfg.preprocess.hop_size,
            ),
            mode="constant",
        )
        # resample the speech to 16k for feature extraction
        if self.cfg.preprocess.sample_rate != 16000:
            speech_16k_prompt0 = librosa.resample(
                speech_prompt0, orig_sr=self.cfg.preprocess.sample_rate, target_sr=16000
            )
        else:
            speech_16k_prompt0 = speech_prompt0

        inputs_prompt0 = self.processor(speech_16k_prompt0, sampling_rate=16000)

        input_features_prompt0 = inputs_prompt0["input_features"][0]
        attention_mask_prompt0 = inputs_prompt0["attention_mask"][0]

        # get speech mask
        speech_frames = len(speech) // self.cfg.preprocess.hop_size
        mask = np.ones(speech_frames)

        speech_frames_prompt0 = len(speech_prompt0) // self.cfg.preprocess.hop_size
        mask_prompt0 = np.ones(speech_frames_prompt0)

        del speech, speech_16k, speech_prompt0, speech_16k_prompt0

        lang_id = self.index2lang[idx]
        phone_id = self.wav_path_index2phoneid[idx]
        phone_id = torch.tensor(phone_id, dtype=torch.long)
        phone_mask = np.ones(len(phone_id))

        single_feature = dict()

        spk_id = self.wav_path_index2spkid[idx]

        single_feature.update({"spk_id": spk_id})
        single_feature.update({"lang_id": lang_id})

        single_feature.update({"phone_id": phone_id})
        single_feature.update({"phone_mask": phone_mask})

        single_feature.update(
            {
                "input_features": input_features,
                "attention_mask": attention_mask,
                "mask": mask,
                "input_features_prompt0": input_features_prompt0,
                "attention_mask_prompt0": attention_mask_prompt0,
                "mask_prompt0": mask_prompt0,
            }
        )

        return single_feature


class T2SCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        for key in batch[0].keys():
            if "input_features" in key:
                packed_batch_features[key] = pad_sequence(
                    [
                        (
                            utt[key].float()
                            if isinstance(utt[key], torch.Tensor)
                            else torch.tensor(utt[key]).float()
                        )
                        for utt in batch
                    ],
                    batch_first=True,
                )
            if "attention_mask" in key:
                packed_batch_features[key] = pad_sequence(
                    [
                        (
                            utt[key].float()
                            if isinstance(utt[key], torch.Tensor)
                            else torch.tensor(utt[key]).float()
                        )
                        for utt in batch
                    ],
                    batch_first=True,
                )
            if "mask" in key:
                packed_batch_features[key] = pad_sequence(
                    [
                        (
                            utt[key].long()
                            if isinstance(utt[key], torch.Tensor)
                            else torch.tensor(utt[key]).long()
                        )
                        for utt in batch
                    ],
                    batch_first=True,
                )
            if "semantic_code" in key:
                packed_batch_features[key] = pad_sequence(
                    [
                        (
                            utt[key].float()
                            if isinstance(utt[key], torch.Tensor)
                            else torch.tensor(utt[key]).float()
                        )
                        for utt in batch
                    ],
                    batch_first=True,
                )
            if key == "phone_id":
                packed_batch_features[key] = pad_sequence(
                    [utt[key].long() for utt in batch],
                    batch_first=True,
                    padding_value=1023,  # phone vocab size is 1024
                )
            if key == "phone_mask":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).long() for utt in batch], batch_first=True
                )
            if key == "lang_id":
                packed_batch_features[key] = torch.tensor(
                    [utt[key] for utt in batch]
                ).long()
            if key == "spk_id":
                packed_batch_features[key] = torch.tensor(
                    [utt[key] for utt in batch]
                ).long()
            if key == "spk_emb_input_features":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).float() for utt in batch], batch_first=True
                )
            if key == "spk_emb_attention_mask":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).long() for utt in batch], batch_first=True
                )
            else:
                pass

        return packed_batch_features


class DownsampleWithMask(nn.Module):
    def __init__(self, downsample_factor=2):
        super(DownsampleWithMask, self).__init__()
        self.downsample_factor = downsample_factor

    def forward(self, x, mask):
        # input from numpy.ndarray to torch.Tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32)

        # print(f"################## x size original {x.shape}################################")

        x = x.float()
        x = x.permute(1, 0)  # to (feature_dim, timestep)
        x = x.unsqueeze(1)  # add channel dimension: (timestep, 1, feature_dim)

        if x.size(-1) < self.downsample_factor:
            raise ValueError("Input size must be larger than downsample factor")

        # print(f"################## x size before {x.shape}################################")
        x = F.avg_pool1d(x, kernel_size=self.downsample_factor)
        x = x.squeeze(
            1
        )  # remove channel dimension: (timestep, feature_dim // downsample_factor)
        x = x.long()
        x = x.permute(1, 0)  # to (feature_dim, timestep)

        mask = mask.float()  # convert mask to float for pooling
        mask = mask.unsqueeze(0).unsqueeze(
            0
        )  # add channel dimension: (timestep, 1, feature_dim)

        if mask.size(-1) < self.downsample_factor:
            raise ValueError("Mask size must be larger than downsample factor")

        mask = F.avg_pool1d(
            mask, kernel_size=self.downsample_factor, stride=self.downsample_factor
        )
        mask = mask.squeeze(0).squeeze(
            0
        )  # remove channel dimension: (timestep, feature_dim // downsample_factor)
        mask = (mask >= 0.5).long()  # if average > 0.5 --> 1, else 0

        return x, mask
