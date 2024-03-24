# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import yaml
import copy
from tqdm import tqdm
from torchaudio.compliance import kaldi
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from fairseq import checkpoint_utils
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from utils.io_optim import (
    TorchaudioDataset,
    LibrosaDataset,
    FFmpegDataset,
    collate_batch,
)
import whisper
from modules.wenet_extractor.utils.init_model import init_model
from modules.wenet_extractor.utils.checkpoint import load_checkpoint

"""
    Extractor for content features
    1. whisper
    2. contentvec
    3. wenet
    4. mert

    Pipeline:
        in preprocess.py:
            call extract_utt_content_features() to extract content features for each utterance
            extract_utt_content_features() envelopes the following steps:
                1. load the model (whisper, contentvec, wenet)
                2. extract the content features
                3. save the content features into files
        in svc_dataset.py:
            call offline_align() to align the content features to the given target length

"""

"""
    Extractor Usage:
        1. initialize an instance of extractor
            extractor = WhisperExtractor(cfg)
        2. load the specified model
            extractor.load_model()
        3. extract the content features
            extractor.extract_content(utt) for single utterance
            extractor.extract_content_batch(utts) for batch utterances
        4. save the content features
            extractor.save_feature(utt, content_feature) for single utterance
"""


class AudioPretrainedModelFeaturesExtractor:
    def __init__(self, cfg, extractor_type):
        self.cfg = cfg
        self.extractor_type = extractor_type
        self.model = None
        self.init_for_retrans()

    def init_for_retrans(self):
        target_hop = self.cfg.preprocess.hop_size

        assert self.extractor_type in ["whisper", "contentvec", "wenet"]
        if self.extractor_type == "whisper":
            source_hop = (
                self.cfg.preprocess.whisper_frameshift
                * self.cfg.preprocess.whisper_downsample_rate
                * self.cfg.preprocess.sample_rate
            )
        elif self.extractor_type == "contentvec":
            source_hop = (
                self.cfg.preprocess.contentvec_frameshift
                * self.cfg.preprocess.sample_rate
            )
        elif self.extractor_type == "wenet":
            source_hop = (
                self.cfg.preprocess.wenet_frameshift
                * self.cfg.preprocess.wenet_downsample_rate
                * self.cfg.preprocess.sample_rate
            )
        source_hop = int(source_hop)
        factor = np.gcd(source_hop, target_hop)
        source_hop //= factor
        target_hop //= factor

        self.source_hop = source_hop
        self.target_hop = target_hop

    def offline_resolution_transformation(self, content, target_len):
        """
        args:
            content: (source_len, dim)
            target_len: target length
        return:
            mapped_feature: (target_len, dim)
        """
        source_hop = self.source_hop
        target_hop = self.target_hop

        # (source_len, 256)
        _, width = content.shape
        # slice the content from padded feature
        source_len = min(target_len * target_hop // source_hop + 1, len(content))

        # const ~= target_len * target_hop
        const = source_len * source_hop // target_hop * target_hop

        # (source_len * source_hop, dim)
        up_sampling_feats = np.repeat(content, source_hop, axis=0)
        # (const, dim) -> (const/target_hop, target_hop, dim) -> (const/target_hop, dim)
        down_sampling_feats = np.average(
            up_sampling_feats[:const].reshape(-1, target_hop, width), axis=1
        )

        err = abs(target_len - len(down_sampling_feats))
        if err > 8:
            # err_log_dir is indeterminate
            err_log_dir = os.path.join(
                self.cfg.preprocess.processed_dir, "align_max_err.log"
            )
            try:
                with open(err_log_dir, "r") as f:
                    err_num = int(f.read())
            except:
                with open(err_log_dir, "w") as f:
                    f.write("0")
                err_num = 0
            if err > err_num:
                with open(err_log_dir, "w") as f:
                    f.write(str(err))

        if len(down_sampling_feats) < target_len:
            # (1, dim) -> (err, dim)
            end = down_sampling_feats[-1][None, :].repeat(err, axis=0)
            down_sampling_feats = np.concatenate([down_sampling_feats, end], axis=0)

        # (target_len, dim)
        mapped_feature = down_sampling_feats[:target_len]

        return mapped_feature

    def log_for_ReTrans(self, err):
        err_log_dir = os.path.join(
            self.cfg.preprocess.processed_dir, "align_max_err.log"
        )
        try:
            with open(err_log_dir, "r") as f:
                err_num = int(f.read())
        except:
            with open(err_log_dir, "w") as f:
                f.write("0")
            err_num = 0
        if err > err_num:
            with open(err_log_dir, "w") as f:
                f.write(str(err))

    def ReTrans(self, source_feats, padded_target_len):
        """
        Resolution Transformation for mismatched frames alginment.

        TODO: Merge the offline resolution_transformation into one

        args:
            source_feats: Tensor, (B, padded_source_len, D)
            padded_target_len: int, the maximum target length in a batch
        return:
            mapped_feature: Tensor, (B, padded_target_len, D)
        """
        source_hop = self.source_hop
        target_hop = self.target_hop

        # (B, padded_source_len, D)
        B, padded_source_len, D = source_feats.shape

        # select the valid content from padded feature
        source_len = min(
            padded_target_len * target_hop // source_hop + 1, padded_source_len
        )

        # const ~= padded_target_len * target_hop (padded wav's duration)
        const = source_len * source_hop // target_hop * target_hop

        # (B, padded_source_len, D) -> (B, padded_source_len * source_hop, D) -> (B, const, D)
        up_sampling_feats = torch.repeat_interleave(source_feats, source_hop, dim=1)[
            :, :const
        ]
        # (B, const, D) -> (B, const/target_hop, target_hop, D) -> (B, const/target_hop, D)
        down_sampling_feats = torch.mean(
            up_sampling_feats.reshape(B, -1, target_hop, D), dim=2
        )

        err = abs(padded_target_len - down_sampling_feats.shape[1])
        if err > 8:
            self.log_for_ReTrans(err)

        if down_sampling_feats.shape[1] < padded_target_len:
            # (B, 1, D) -> (B, err, D)
            end = down_sampling_feats[:, -1, :][:, None, :].repeat_interleave(
                err, dim=1
            )
            # -> (B, padded_target_len, D)
            down_sampling_feats = torch.cat([down_sampling_feats, end], dim=1)

        # (B, padded_target_len, D)
        mapped_feature = down_sampling_feats[:, :padded_target_len]
        return mapped_feature

    def get_valid_features(self, utt, content_feature):
        # only keep effective parts
        duration = utt["Duration"]
        if self.extractor_type == "whisper":
            frameshift = (
                self.cfg.preprocess.whisper_frameshift
                * self.cfg.preprocess.whisper_downsample_rate
            )  # 20ms
        elif self.extractor_type == "contentvec":
            frameshift = self.cfg.preprocess.contentvec_frameshift  # 20ms
        elif self.extractor_type == "wenet":
            frameshift = (
                self.cfg.preprocess.wenet_frameshift
                * self.cfg.preprocess.wenet_downsample_rate
            )  # 40ms
        elif self.extractor_type == "mert":
            frameshift = self.cfg.preprocess.mert_frameshift
        else:
            raise NotImplementedError

        # calculate the number of valid frames
        num_frames = int(np.ceil((duration - frameshift) / frameshift)) + 1
        assert (
            len(content_feature.shape) == 2
        ), "content feature shape error, it should be (num_frames, dim)"
        content_feature = content_feature[:num_frames, :]
        return content_feature

    def save_feature(self, utt, content_feature):
        """Save a single utternace to path {cfg.preprocess.processed_dir}

        Args:
            utt (dict): one item in metadata, containing information for one utterance
            content_feature (tensor): content feature of one utterance
        """
        uid = utt["Uid"]
        assert self.extractor_type != None
        out_dir = os.path.join(
            self.cfg.preprocess.processed_dir, utt["Dataset"], self.extractor_type
        )
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, uid + ".npy")

        content_feature = self.get_valid_features(utt, content_feature)
        np.save(save_path, content_feature.cpu().detach().numpy())


class WhisperExtractor(AudioPretrainedModelFeaturesExtractor):
    def __init__(self, config):
        super(WhisperExtractor, self).__init__(config, extractor_type="whisper")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        # load whisper checkpoint
        print("Loading Whisper Model...")

        if "whisper_model_path" in self.cfg.preprocess:
            if os.path.isfile(self.cfg.preprocess.whisper_model_path):
                # "pretrained/whisper/medium.pt"
                download_root = os.path.dirname(self.cfg.preprocess.whisper_model_path)
            elif os.path.isdir(self.cfg.preprocess.whisper_model_path):
                # "pretrained/whisper"
                download_root = self.cfg.preprocess.whisper_model_path
            else:
                # if the path does not exist, download the model to the path
                download_root = self.cfg.preprocess.whisper_model_path
                if download_root.endswith(".pt"):
                    download_root = os.path.dirname(download_root)
        else:
            download_root = None

        model = whisper.load_model(
            self.cfg.preprocess.whisper_model, self.device, download_root
        )
        if torch.cuda.is_available():
            print("Using GPU...\n")
            model = model.cuda()
        else:
            print("Using CPU...\n")

        self.model = model.eval()

    def extract_content_features(self, wavs):
        """extract content features from a batch of dataloader
        Args:
            wavs: tensor (batch_size, T)
        """
        # wavs: (batch, max_len)
        wavs = whisper.pad_or_trim(wavs)
        # batch_mel: (batch, 80, 3000)
        batch_mel = whisper.log_mel_spectrogram(wavs, device=self.model.device)
        with torch.no_grad():
            # (batch, 1500, 1024)
            features = self.model.embed_audio(batch_mel)
        return features


class ContentvecExtractor(AudioPretrainedModelFeaturesExtractor):
    def __init__(self, cfg):
        super(ContentvecExtractor, self).__init__(cfg, extractor_type="contentvec")

    def load_model(self):
        assert self.model == None
        # Load model
        ckpt_path = self.cfg.preprocess.contentvec_file
        print("Load Contentvec Model...")

        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path],
            suffix="",
        )
        model = models[0]
        model.eval()

        if torch.cuda.is_available():
            # print("Using GPU...\n")
            model = model.cuda()

        self.model = model

    def extract_content_features(self, wavs):
        """extract content features from a batch of dataloader
        Args:
            wavs: tensor (batch, T)
        """
        device = next(self.model.parameters()).device
        wavs = wavs.to(device)  # (batch, max_len)
        padding_mask = torch.eq(wavs, torch.zeros_like(wavs)).to(device)
        with torch.no_grad():
            logits = self.model.extract_features(
                source=wavs, padding_mask=padding_mask, output_layer=12
            )
            # feats: (batch, T, 256)
            feats = self.model.final_proj(logits[0])
        return feats


class WenetExtractor(AudioPretrainedModelFeaturesExtractor):
    def __init__(self, config):
        super(WenetExtractor, self).__init__(config, extractor_type="wenet")

    def load_model(self):
        wenet_cfg = self.cfg.preprocess.wenet_config
        wenet_model_path = self.cfg.preprocess.wenet_model_path
        # load Wenet config
        with open(wenet_cfg, "r") as w:
            wenet_configs = yaml.load(w, Loader=yaml.FullLoader)
        self.extract_conf = copy.deepcopy(wenet_configs["dataset_conf"])
        print("Loading Wenet Model...")
        self.model = init_model(wenet_configs)
        load_checkpoint(self.model, wenet_model_path)

        if torch.cuda.is_available():
            print("Using GPU...\n")
            self.model = self.model.cuda()
        else:
            print("Using CPU...\n")

        self.model = self.model.eval()

    def extract_content_features(self, wavs, lens):
        """extract content features from a batch of dataloader
        Args:
            wavs: tensor, whose shape is (B, T)
            lens: list
        """
        feats_list = []
        lengths_list = []

        device = next(self.model.parameters()).device
        # Extract fbank/mfcc features by kaldi
        assert self.extract_conf is not None, "load model first!"
        feats_type = self.extract_conf.get("feats_type", "fbank")
        assert feats_type in ["fbank", "mfcc"]

        for idx, wav in enumerate(wavs):
            # wav: (T)
            wav = wav[: lens[idx]].to(device)

            # pad one frame to compensate for the frame cut off after feature extraction
            pad_tensor = torch.zeros(160, device=wav.device)
            wav = torch.cat((wav, pad_tensor), dim=-1)
            wav *= 1 << 15

            wav = wav.unsqueeze(0)  # (T) -> (1, T)
            if feats_type == "fbank":
                fbank_conf = self.extract_conf.get("fbank_conf", {})
                feat = kaldi.fbank(
                    wav,
                    sample_frequency=16000,
                    num_mel_bins=fbank_conf["num_mel_bins"],
                    frame_length=fbank_conf["frame_length"],
                    frame_shift=fbank_conf["frame_shift"],
                    dither=fbank_conf["dither"],
                )
            elif feats_type == "mfcc":
                mfcc_conf = self.extract_conf.get("mfcc", {})
                feat = kaldi.mfcc(
                    wav,
                    sample_frequency=16000,
                    num_mel_bins=mfcc_conf["num_mel_bins"],
                    frame_length=mfcc_conf["frame_length"],
                    frame_shift=mfcc_conf["frame_shift"],
                    dither=mfcc_conf["dither"],
                    num_ceps=mfcc_conf.get("num_ceps", 40),
                    high_freq=mfcc_conf.get("high_freq", 0.0),
                    low_freq=mfcc_conf.get("low_freq", 20.0),
                )
            feats_list.append(feat)
            lengths_list.append(feat.shape[0])

        feats_lengths = torch.tensor(lengths_list, dtype=torch.int32).to(device)
        feats_tensor = pad_sequence(feats_list, batch_first=True).to(
            device
        )  # (batch, len, 80)

        features = self.model.encoder_extractor(
            feats_tensor,
            feats_lengths,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
            simulate_streaming=False,
        )
        return features


class MertExtractor(AudioPretrainedModelFeaturesExtractor):
    def __init__(self, cfg):
        super(MertExtractor, self).__init__(cfg, extractor_type="mert")
        self.preprocessor = None

    def load_model(self):
        assert self.model == None
        assert self.preprocessor == None

        print("Loading MERT Model: ...", self.cfg.preprocess.mert_model)

        model_name = self.cfg.preprocess.mert_model
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if torch.cuda.is_available():
            model = model.cuda()
        preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = model
        self.preprocessor = preprocessor

    def extract_content_features(self, wavs):
        """extract content features from a batch of dataloader
        Args:
            wavs: tensor (batch, T)
        """
        with torch.no_grad():
            sample_rate = self.preprocessor.sampling_rate
            device = next(self.model.parameters()).device
            assert (
                sample_rate == self.cfg.preprocess.mert_sample_rate
            ), "mert sample rate mismatch, expected {}, got {}".format(
                self.cfg.preprocess.mert_sample_rate, sample_rate
            )
            mert_features = []
            # wav: (len)
            for wav in wavs:
                # {input_values: tensor, attention_mask: tensor}
                inputs = self.preprocessor(
                    wavs, sampling_rate=sample_rate, return_tensors="pt"
                ).to(device)

                outputs = self.model(**inputs, output_hidden_states=True)
                # (25 layers, time steps, 1024 feature_dim)
                all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
                # (1, frame_len, 1024) -> (frame_len, 1024)
                feature = outputs.hidden_states[
                    self.cfg.preprocess.mert_feature_layer
                ].squeeze(0)
                mert_features.append(feature)

        return mert_features


def extract_utt_content_features_dataloader(cfg, metadata, num_workers):
    dataset_name = metadata[0]["Dataset"]
    with torch.no_grad():
        if cfg.preprocess.extract_whisper_feature:
            feat_dir = os.path.join(
                cfg.preprocess.processed_dir, dataset_name, "whisper"
            )
            os.makedirs(feat_dir, exist_ok=True)
            feat_files_num = len(os.listdir(feat_dir))

            if feat_files_num != len(metadata):
                whisper_waveforms = FFmpegDataset(
                    cfg,
                    dataset_name,
                    cfg.preprocess.whisper_sample_rate,
                    metadata=metadata,
                )
                data_loader = DataLoader(
                    whisper_waveforms,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=cfg.preprocess.pin_memory,
                    batch_size=cfg.preprocess.content_feature_batch_size,
                    collate_fn=collate_batch,
                    drop_last=False,
                )
                extractor = WhisperExtractor(cfg)
                extractor.load_model()
                for batch_idx, items in enumerate(tqdm(data_loader)):
                    _metadata, wavs, lens = items

                    batch_content_features = extractor.extract_content_features(wavs)
                    for index, utt in enumerate(_metadata):
                        extractor.save_feature(utt, batch_content_features[index])

        if cfg.preprocess.extract_contentvec_feature:
            feat_dir = os.path.join(
                cfg.preprocess.processed_dir, dataset_name, "contentvec"
            )
            os.makedirs(feat_dir, exist_ok=True)
            feat_files_num = len(os.listdir(feat_dir))

            if feat_files_num != len(metadata):
                contentvec_waveforms = LibrosaDataset(
                    cfg,
                    dataset_name,
                    cfg.preprocess.contentvec_sample_rate,
                    metadata=metadata,
                )
                data_loader = DataLoader(
                    contentvec_waveforms,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=cfg.preprocess.pin_memory,
                    batch_size=cfg.preprocess.content_feature_batch_size,
                    collate_fn=collate_batch,
                    drop_last=False,
                )
                extractor = ContentvecExtractor(cfg)
                extractor.load_model()
                for batch_idx, items in enumerate(tqdm(data_loader)):
                    _metadata, wavs, lens = items

                    batch_content_features = extractor.extract_content_features(wavs)
                    for index, utt in enumerate(_metadata):
                        extractor.save_feature(utt, batch_content_features[index])

        if cfg.preprocess.extract_wenet_feature:
            feat_dir = os.path.join(cfg.preprocess.processed_dir, dataset_name, "wenet")
            os.makedirs(feat_dir, exist_ok=True)
            feat_files_num = len(os.listdir(feat_dir))

            if feat_files_num != len(metadata):
                wenet_waveforms = TorchaudioDataset(
                    cfg,
                    dataset_name,
                    cfg.preprocess.wenet_sample_rate,
                    metadata=metadata,
                )
                data_loader = DataLoader(
                    wenet_waveforms,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=cfg.preprocess.pin_memory,
                    batch_size=cfg.preprocess.content_feature_batch_size,
                    collate_fn=collate_batch,
                    drop_last=False,
                )
                extractor = WenetExtractor(cfg)
                extractor.load_model()
                for batch_idx, items in enumerate(tqdm(data_loader)):
                    _metadata, wavs, lens = items

                    batch_content_features = extractor.extract_content_features(
                        wavs,
                        lens,
                    )
                    for index, utt in enumerate(_metadata):
                        extractor.save_feature(utt, batch_content_features[index])

        if cfg.preprocess.extract_mert_feature:
            feat_dir = os.path.join(cfg.preprocess.processed_dir, dataset_name, "mert")
            os.makedirs(feat_dir, exist_ok=True)
            feat_files_num = len(os.listdir(feat_dir))

            if feat_files_num != len(metadata):
                mert_waveforms = TorchaudioDataset(
                    cfg,
                    dataset_name,
                    cfg.preprocess.mert_sample_rate,
                    metadata=metadata,
                )
                data_loader = DataLoader(
                    mert_waveforms,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=cfg.preprocess.pin_memory,
                    batch_size=cfg.preprocess.content_feature_batch_size,
                    collate_fn=collate_batch,
                    drop_last=False,
                )
                extractor = MertExtractor(cfg)
                extractor.load_model()
                for batch_idx, items in enumerate(tqdm(data_loader)):
                    _metadata, wavs, lens = items

                    batch_content_features = extractor.extract_content_features(wavs)
                    for index, utt in enumerate(_metadata):
                        extractor.save_feature(utt, batch_content_features[index])
