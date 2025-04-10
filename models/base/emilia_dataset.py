# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import librosa
import torch
import json
import numpy as np
import logging
import pickle
import os
from pathlib import Path
import random


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


MNT_PATH = "[Please fill out your emilia data root path]"
CACHE_PATH = "[Please fill out your emilia cache path]"


class EmiliaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_type="path",
        cfg=None,
    ):  # 'path' or 'meta'

        assert cfg is not None

        self.cache_type = cache_type
        self.cfg = cfg

        self.dataset_ratio_dict = self.cfg.dataset
        self.emilia_ratio = self.dataset_ratio_dict["emilia"]

        self.json_paths = []
        self.wav_paths = []
        self.mnt_path = MNT_PATH

        self.language_list = ["zh", "en"]  # Data language list
        self.wav_path_index2duration = []
        self.wav_path_index2phonelen = []
        self.index2num_frames = []

        self.json_path2meta = {}
        self.json2filtered_idx = {}

        self.cache_folder = CACHE_PATH
        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)

        self.wav_paths_cache = os.path.join(self.cache_folder, "wav_paths_cache.pkl")
        self.json_paths_cache = os.path.join(self.cache_folder, "json_paths_cache.pkl")
        self.duration_cache = os.path.join(self.cache_folder, "duration_cache.pkl")
        self.phone_count_cache = os.path.join(
            self.cache_folder, "phone_count_cache.pkl"
        )
        self.json_path2meta_cache = os.path.join(
            self.cache_folder, "json_path2meta.pkl"
        )

        if cache_type == "path":
            if (
                os.path.exists(self.wav_paths_cache)
                and os.path.exists(self.json_paths_cache)
                and os.path.exists(self.duration_cache)
                and os.path.exists(self.phone_count_cache)
            ):
                self.load_cached_paths()
        else:
            logger.info("Incorrect cache loading way")
            exit()

        if cache_type == "meta":
            if os.path.exists(self.json_path2meta_cache):
                self.load_path2meta()
            else:
                self.get_jsoncache_multiprocess(pool_size=8)

        self.num_frame_indices = np.array(
            sorted(
                range(len(self.index2num_frames)),
                key=lambda k: self.index2num_frames[k],
            )
        )

        self.duration_setting = {"min": 3, "max": 30}
        if hasattr(self.cfg.preprocess, "min_dur"):
            self.duration_setting["min"] = self.cfg.preprocess.min_dur
        if hasattr(self.cfg.preprocess, "max_dur"):
            self.duration_setting["max"] = self.cfg.preprocess.max_dur

    def load_cached_paths(self):
        logger.info("Loaded paths from cache files")
        with open(self.wav_paths_cache, "rb") as f:
            all_wav_paths = pickle.load(f)
        with open(self.json_paths_cache, "rb") as f:
            all_json_paths = pickle.load(f)

        # Select part of data according to emilia_ratio
        if self.emilia_ratio < 1.0:
            total_samples = len(all_wav_paths)
            num_samples = int(total_samples * self.emilia_ratio)
            selected_indices = random.sample(range(total_samples), num_samples)

            self.wav_paths = [all_wav_paths[i] for i in selected_indices]

            # TODO: check what does json_paths do.
            # self.json_paths = [all_json_paths[i] for i in selected_indices]
            self.json_paths = []

            if self.cache_type == "path":
                with open(self.duration_cache, "rb") as f:
                    all_durations = pickle.load(f)
                with open(self.phone_count_cache, "rb") as f:
                    all_phone_counts = pickle.load(f)

                self.wav_path_index2duration = [
                    all_durations[i] for i in selected_indices
                ]
                self.wav_path_index2phonelen = [
                    all_phone_counts[i] for i in selected_indices
                ]
        else:
            assert self.emilia_ratio == 1

            self.wav_paths = all_wav_paths
            self.json_paths = all_json_paths
            if self.cache_type == "path":
                with open(self.duration_cache, "rb") as f:
                    self.wav_path_index2duration = pickle.load(f)
                with open(self.phone_count_cache, "rb") as f:
                    self.wav_path_index2phonelen = pickle.load(f)

        # Calculate the number of frames
        if self.cache_type == "path":
            self.index2num_frames = []
            for duration, phone_count in zip(
                self.wav_path_index2duration, self.wav_path_index2phonelen
            ):
                self.index2num_frames.append(duration * 50 + phone_count)

        logger.info("All Emilia paths got successfully, ratio: %f" % self.emilia_ratio)
        logger.info(
            "Number of wavs: %d, Number of jsons: %d"
            % (len(self.wav_paths), len(self.json_paths))
        )

    def save_cached_paths(self):
        with open(self.wav_paths_cache, "wb") as f:
            pickle.dump(self.wav_paths, f)
        with open(self.json_paths_cache, "wb") as f:
            pickle.dump(self.json_paths, f)
        if self.cache_type == "path":
            with open(self.duration_cache, "wb") as f:
                pickle.dump(self.wav_path_index2duration, f)
            with open(self.phone_count_cache, "wb") as f:
                pickle.dump(self.wav_path_index2phonelen, f)
        logger.info("Saved paths to cache files")

    # Load JSON data from a compressed GZIP file
    def load_compressed_json(self, filename):
        import gzip

        with gzip.open(filename, "rt", encoding="utf-8") as f:
            return json.load(f)

    def get_phone_count_and_duration(self, meta, idx_list):
        new_meta = {}
        if meta[0]["language"] not in self.language_list:
            new_meta["0"] = meta[0]
            return new_meta
        text_list = []
        for i in idx_list:
            text_list.append(meta[i]["text"])
        token_id = self.g2p(text_list, meta[0]["language"])[1]
        for i, token in zip(idx_list, token_id):
            nm = {}
            nm["language"] = meta[i]["language"]
            nm["phone_id"] = token
            nm["phone_count"] = len(token)
            nm["duration"] = meta[i]["end"] - meta[i]["start"]
            new_meta[str(i)] = nm
        del meta
        return new_meta

    # Only 'meta' cache type use
    def load_path2meta(self):
        logger.info("Loaded meta from cache files")
        self.json_path2meta = pickle.load(open(self.json_path2meta_cache, "rb"))
        for path in self.wav_paths:
            duration = self.get_meta_from_wav_path(path)["duration"]
            phone_count = self.get_meta_from_wav_path(path)["phone_count"]
            self.wav_path_index2duration.append(duration)
            self.wav_path_index2phonelen.append(phone_count)
            self.index2num_frames.append(duration * 50)
            # self.index2num_frames.append(duration * self.cfg.preprocess.sample_rate)

    def get_meta_from_wav_path(self, wav_path):
        wav_path = wav_path.replace("wav_new/", "")
        index = int(wav_path.split("_")[-1].split(".")[0])
        audio_name = "_".join(wav_path.split("/")[-1].split("_")[:-1])
        dir_name = "/".join(wav_path.split("/")[:-1])
        json_name = audio_name + "_fixzh.json"
        json_path = dir_name + "/" + json_name
        meta = None
        if self.cache_type == "meta":
            meta = self.json_path2meta[json_path][str(index)]
            return meta
        elif self.cache_type == "path":
            try:
                buffer = json_path.replace("_fixzh", "")
                if "/MLS/" in json_path:
                    with open(buffer, "r") as f:
                        meta = json.load(f)[os.path.basename(wav_path)]
                else:
                    with open(buffer, "r") as f:
                        meta = json.load(f)[index]

            except Exception as e:
                logger.info("Error json: {} error: {}".format(json_path, e))
        del index, audio_name, dir_name, json_name, json_path
        return meta

    def __len__(self):
        return self.wav_paths.__len__()

    def get_num_frames(self, index):
        return self.wav_path_index2duration[index] * 50

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        file_bytes = None
        try:
            # wav_path = MNT_PATH + "wav_new/" + wav_path.replace("_new", "")
            wav_path = self.mnt_path + wav_path
            file_bytes = wav_path
        except:
            logger.info("Get data from failed. Get another.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            del position
            return self.__getitem__(random_index)

        meta = self.get_meta_from_wav_path(wav_path)
        if file_bytes is not None and meta is not None:
            buffer = file_bytes
            try:
                speech, sr = librosa.load(buffer, sr=self.cfg.preprocess.sample_rate)
                if (
                    len(speech)
                    > self.duration_setting["max"] * self.cfg.preprocess.sample_rate
                ):
                    position = np.where(self.num_frame_indices == idx)[0][0]
                    random_index = np.random.choice(self.num_frame_indices[:position])
                    del position
                    return self.__getitem__(random_index)
            except:
                logger.info("Failed to load file. Get another.")
                position = np.where(self.num_frame_indices == idx)[0][0]
                random_index = np.random.choice(self.num_frame_indices[:position])
                del position
                return self.__getitem__(random_index)

            single_feature = dict()

            # pad the speech to the multiple of hop_size
            speech = np.pad(
                speech,
                (
                    0,
                    self.cfg.preprocess.hop_size
                    - len(speech) % self.cfg.preprocess.hop_size,
                ),
                mode="constant",
            )

            # get speech mask
            speech_frames = len(speech) // self.cfg.preprocess.hop_size
            mask = np.ones(speech_frames)

            single_feature.update(
                {
                    "speech": speech,
                    "mask": mask,
                }
            )

            return single_feature

        else:
            logger.info("Failed to get file after retries.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            del position
            return self.__getitem__(random_index)
