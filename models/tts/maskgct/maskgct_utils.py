# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import pickle
import math
import json
import accelerate
import safetensors
from utils.util import load_config
from tqdm import tqdm

from models.codec.kmeans.repcodec_model import RepCodec
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from transformers import Wav2Vec2BertModel

from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p

from transformers import SeamlessM4TFeatureExtractor


def g2p_(text, language):
    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return g2p(text, sentence=None, language=language)


def build_t2s_model(cfg, device):
    t2s_model = MaskGCT_T2S(cfg=cfg)
    t2s_model.eval()
    t2s_model.to(device)
    return t2s_model


def build_s2a_model(cfg, device):
    soundstorm_model = MaskGCT_S2A(cfg=cfg)
    soundstorm_model.eval()
    soundstorm_model.to(device)
    return soundstorm_model


def build_semantic_model(device):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)
    stat_mean_var = torch.load("./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt")
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg, device):
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    semantic_codec.to(device)
    return semantic_codec


def build_acoustic_codec(cfg, device):
    codec_encoder = CodecEncoder(cfg=cfg.encoder)
    codec_decoder = CodecDecoder(cfg=cfg.decoder)
    codec_encoder.eval()
    codec_decoder.eval()
    codec_encoder.to(device)
    codec_decoder.to(device)
    return codec_encoder, codec_decoder


class MaskGCT_Inference_Pipeline:
    def __init__(
        self,
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    ):
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.codec_encoder = codec_encoder
        self.codec_decoder = codec_decoder
        self.t2s_model = t2s_model
        self.s2a_model_1layer = s2a_model_1layer
        self.s2a_model_full = s2a_model_full
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.device = device

    @torch.no_grad()
    def extract_features(self, speech):
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]
        return input_features, attention_mask

    @torch.no_grad()
    def extract_semantic_code(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        semantic_code, rec_feat = self.semantic_codec.quantize(feat)  # (B, T)
        return semantic_code, rec_feat

    @torch.no_grad()
    def extract_acoustic_code(self, speech):
        vq_emb = self.codec_encoder(speech.unsqueeze(1))
        _, vq, _, _, _ = self.codec_decoder.quantizer(vq_emb)
        acoustic_code = vq.permute(1, 2, 0)
        return acoustic_code

    @torch.no_grad()
    def text2semantic(
        self,
        prompt_speech,
        prompt_text,
        prompt_language,
        target_text,
        target_language,
        target_len=None,
        n_timesteps=50,
        cfg=2.5,
        rescale_cfg=0.75,
    ):
        prompt_phone_id = g2p_(prompt_text, prompt_language)[1]
        target_phone_id = g2p_(target_text, target_language)[1]

        if target_len is None:
            target_len = int(
                (len(prompt_speech) * len(target_phone_id) / len(prompt_phone_id))
                / 16000
                * 50
            )
        else:
            target_len = int(target_len * 50)

        prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(
            self.device
        )
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(
            self.device
        )

        phone_id = torch.cat([prompt_phone_id, target_phone_id])

        input_features, attention_mask = self.extract_features(prompt_speech)
        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        semantic_code, _ = self.extract_semantic_code(input_features, attention_mask)

        predict_semantic = self.t2s_model.reverse_diffusion(
            semantic_code[:, :],
            target_len,
            phone_id.unsqueeze(0),
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        print("predict semantic shape", predict_semantic.shape)

        combine_semantic_code = torch.cat(
            [semantic_code[:, :], predict_semantic], dim=-1
        )
        prompt_semantic_code = semantic_code

        return combine_semantic_code, prompt_semantic_code

    @torch.no_grad()
    def semantic2acoustic(
        self,
        combine_semantic_code,
        acoustic_code,
        n_timesteps=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        cfg=2.5,
        rescale_cfg=0.75,
    ):
        semantic_code = combine_semantic_code

        cond = self.s2a_model_1layer.cond_emb(semantic_code)
        prompt = acoustic_code[:, :, :]
        predict_1layer = self.s2a_model_1layer.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps[:1],
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        cond = self.s2a_model_full.cond_emb(semantic_code)
        prompt = acoustic_code[:, :, :]
        predict_full = self.s2a_model_full.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            gt_code=predict_1layer,
        )

        vq_emb = self.codec_decoder.vq2emb(
            predict_full.permute(2, 0, 1), n_quantizers=12
        )
        recovered_audio = self.codec_decoder(vq_emb)
        prompt_vq_emb = self.codec_decoder.vq2emb(
            prompt.permute(2, 0, 1), n_quantizers=12
        )
        recovered_prompt_audio = self.codec_decoder(prompt_vq_emb)
        recovered_prompt_audio = recovered_prompt_audio[0][0].cpu().numpy()
        recovered_audio = recovered_audio[0][0].cpu().numpy()
        combine_audio = np.concatenate([recovered_prompt_audio, recovered_audio])

        return combine_audio, recovered_audio

    def maskgct_inference(
        self,
        prompt_speech_path,
        prompt_text,
        target_text,
        language="en",
        target_language="en",
        target_len=None,
        n_timesteps=25,
        cfg=2.5,
        rescale_cfg=0.75,
        n_timesteps_s2a=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        cfg_s2a=2.5,
        rescale_cfg_s2a=0.75,
    ):
        speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]
        speech = librosa.load(prompt_speech_path, sr=24000)[0]

        combine_semantic_code, _ = self.text2semantic(
            speech_16k,
            prompt_text,
            language,
            target_text,
            target_language,
            target_len,
            n_timesteps,
            cfg,
            rescale_cfg,
        )
        acoustic_code = self.extract_acoustic_code(
            torch.tensor(speech).unsqueeze(0).to(self.device)
        )
        _, recovered_audio = self.semantic2acoustic(
            combine_semantic_code,
            acoustic_code,
            n_timesteps=n_timesteps_s2a,
            cfg=cfg_s2a,
            rescale_cfg=rescale_cfg_s2a,
        )

        return recovered_audio
