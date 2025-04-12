# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import librosa
import torch
from torch.nn.utils.rnn import pad_sequence

from models.base.emilia_dataset import EmiliaDataset, WarningFilter


filter = WarningFilter()
logging.getLogger("phonemizer").addFilter(filter)
logging.getLogger("qcloud_cos.cos_client").addFilter(filter)
logging.getLogger("jieba").addFilter(filter)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VCEmiliaDataset(EmiliaDataset):
    def __init__(self, cfg):
        super(VCEmiliaDataset, self).__init__(cfg=cfg)

        self.sample_rate = self.cfg.preprocess.sample_rate

        # Audio pretrained models' sample rates
        self.all_sample_rates = {self.sample_rate}
        if hasattr(self.cfg.model, "cond_sample_rate"):
            self.all_sample_rates.add(self.cfg.model.cond_sample_rate)

        self.load_phone = getattr(self.cfg.preprocess, "load_phone", False)
        self.load_wav_path = getattr(self.cfg.preprocess, "load_wav_path", False)
        self.load_semantic_features = getattr(
            self.cfg.preprocess, "load_semantic_features", False
        )
        self.load_chromagram = getattr(self.cfg.preprocess, "load_chromagram", False)

        if self.load_semantic_features:
            from transformers import SeamlessM4TFeatureExtractor

            self.semantic_model_processor = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0"
            )

    def g2p(self, text, language):
        from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p

        if language in ["zh", "en"]:
            return chn_eng_g2p(text)
        else:
            return g2p(text, sentence=None, language=language)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        file_bytes = None
        try:
            # wav_path = MNT_PATH + "wav_new/" + wav_path.replace("_new", "")
            wav_path = self.mnt_path + wav_path
            file_bytes = wav_path
        except:
            logger.info(f"Get data from {wav_path} failed. Get another.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            del position
            return self.__getitem__(random_index)

        meta = self.get_meta_from_wav_path(wav_path)
        if file_bytes is not None and meta is not None:
            buffer = file_bytes
            try:
                speech, _ = librosa.load(buffer, sr=self.sample_rate)
                if len(speech) > self.duration_setting["max"] * self.sample_rate:
                    position = np.where(self.num_frame_indices == idx)[0][0]
                    random_index = np.random.choice(self.num_frame_indices[:position])
                    del position
                    return self.__getitem__(random_index)
            except:
                logger.info(f"Failed to load file {wav_path}. Get another.")
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

            # For all the sample rates
            for tgt_sr in self.all_sample_rates:
                if tgt_sr != self.sample_rate:
                    assert tgt_sr < self.sample_rate
                    tgt_speech = librosa.resample(
                        speech, orig_sr=self.sample_rate, target_sr=tgt_sr
                    )
                else:
                    tgt_speech = speech
                single_feature.update(
                    {
                        f"wav_{tgt_sr}": tgt_speech,
                        f"wav_{tgt_sr}_len": len(tgt_speech),
                    }
                )

            # [Note] Mask is (n_frames,) but not (T,)
            speech_frames = len(speech) // self.cfg.preprocess.hop_size
            mask = np.ones(speech_frames)

            single_feature.update(
                {
                    "wav": speech,
                    "wav_len": len(speech),
                    "mask": mask,
                }
            )

            ## Load Semantic Model Input Features ##
            if self.load_semantic_features:
                speech_16k = single_feature["wav_16000"]
                inputs = self.semantic_model_processor(speech_16k, sampling_rate=16000)
                input_features = inputs["input_features"][0]
                attention_mask = inputs["attention_mask"][0]

                single_feature.update(
                    {
                        "semantic_model_input_features": input_features,
                        "semantic_model_attention_mask": attention_mask,
                    }
                )

            if self.load_wav_path:
                single_feature.update({"wav_path": wav_path})

            if not self.load_phone:
                return single_feature

            ## Load phone using G2P ##
            try:
                phone_id = (
                    self.g2p(meta["text"], meta["language"])[1]
                    if self.cache_type == "path"
                    else meta["phone_id"]
                )
                if len(phone_id) > 512:
                    raise Exception("too long phone seq")
            except Exception as e:
                print(e)
                print(f"Loading phone failed for {wav_path}")
                print(meta["text"], meta["language"])
                position = np.where(self.num_frame_indices == idx)[0][0]
                random_index = np.random.choice(self.num_frame_indices[:position])
                del position
                return self.__getitem__(random_index)
            if len(phone_id) >= speech_frames:
                position = np.where(self.num_frame_indices == idx)[0][0]
                random_index = np.random.choice(self.num_frame_indices[:position])
                del position
                return self.__getitem__(random_index)

            phone_id = torch.tensor(np.array(phone_id), dtype=torch.long)
            phone_mask = np.ones(len(phone_id))

            single_feature.update({"phone_id": phone_id, "phone_mask": phone_mask})
            return single_feature

        else:
            logger.info("Failed to get file after retries.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            del position
            return self.__getitem__(random_index)


class VCCollator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        """
        VCEmiliaDataset.__getitem__:
            wav: (T,)
            wav_len: int
            mask: (n_frames,)

            wav_{sr}: (T,)
            wav_{sr}_len: int

            phone_id: (n_phones,)
            phone_mask: (n_phones,)

        Returns:
            wav: (B, T), torch.float32
            wav_len: (B), torch.long
            mask: (B, n_frames), torch.float32

            wav_{sr}: (B, T)
            wav_{sr}_len: (B), torch.long

            phone_id: (B, n_phones), torch.long
            phone_mask: (B, n_phones), torch.float32
        """

        packed_batch_features = dict()

        for key in batch[0].keys():
            if "_len" in key:
                packed_batch_features[key] = torch.LongTensor([b[key] for b in batch])
            elif key == "phone_id":
                packed_batch_features[key] = pad_sequence(
                    [utt[key].long() for utt in batch],
                    batch_first=True,
                    padding_value=1023,  # phone vocab size is 1024
                )
            elif key == "wav_path":
                packed_batch_features[key] = [b[key] for b in batch]
            else:
                packed_batch_features[key] = pad_sequence(
                    [torch.as_tensor(b[key]) for b in batch],
                    batch_first=True,
                    padding_value=0.0,
                )
        return packed_batch_features
