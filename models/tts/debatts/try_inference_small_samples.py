# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

os.chdir("./models/tts/debatts")
sys.path.append("./models/tts/debatts")
from utils.g2p_new.g2p_new import new_g2p

from transformers import Wav2Vec2Model
from cgitb import text
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from IPython.display import Audio
import matplotlib.pyplot as plt
import soundfile as sf
import pickle
import math
import json
import accelerate
from IPython.display import Audio

from models.codec.kmeans.kmeans_model import KMeans, KMeansEMA
from models.codec.kmeans.repcodec_model import RepCodec
from models.tts.soundstorm.soundstorm_model import SoundStorm
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from transformers import Wav2Vec2BertModel
import safetensors
from utils.util import load_config
from tqdm import tqdm

from transformers import SeamlessM4TFeatureExtractor

processor = SeamlessM4TFeatureExtractor.from_pretrained("./ckpt/w2v-bert-2")

from transformers import AutoProcessor, AutoModel

from models.tts.text2semantic.t2s_model import T2SLlama
from models.tts.text2semantic.t2s_model_new import T2SLlama_new
from models.tts.text2semantic.t2s_sft_dataset_new import DownsampleWithMask


def new_g2p_(text, language):
    return new_g2p(text, language)


def build_t2s_model_new(cfg, device):
    t2s_model = T2SLlama_new(
        phone_vocab_size=1024,
        target_vocab_size=8192,
        hidden_size=2048,
        intermediate_size=8192,
        pad_token_id=9216,
        bos_target_id=9217,
        eos_target_id=9218,
        bos_phone_id=9219,
        eos_phone_id=9220,
        bos_prompt0_id=9221,
        eos_prompt0_id=9222,
        use_lang_emb=False,
    )
    t2s_model.eval()
    t2s_model.to(device)
    t2s_model.half()
    return t2s_model


def build_soundstorm(cfg, device):
    soundstorm_model = SoundStorm(cfg=cfg.model.soundstorm)
    soundstorm_model.eval()
    soundstorm_model.to(device)
    return soundstorm_model


def build_kmeans_model(cfg, device):
    if cfg.model.kmeans.type == "kmeans":
        kmeans_model = KMeans(cfg=cfg.model.kmeans.kmeans)
    elif cfg.model.kmeans.type == "kmeans_ema":
        kmeans_model = KMeansEMA(cfg=cfg.model.kmeans.kmeans)
    elif cfg.model.kmeans.type == "repcodec":
        kmeans_model = RepCodec(cfg=cfg.model.kmeans.repcodec)
    kmeans_model.eval()
    pretrained_path = cfg.model.kmeans.pretrained_path
    if ".bin" in pretrained_path:
        kmeans_model.load_state_dict(torch.load(pretrained_path))
    elif ".safetensors" in pretrained_path:
        safetensors.torch.load_model(kmeans_model, pretrained_path)
    kmeans_model.to(device)
    return kmeans_model


def build_semantic_model(cfg, device):
    semantic_model = Wav2Vec2BertModel.from_pretrained("./w2v-bert-2")
    semantic_model.eval()
    semantic_model.to(device)

    layer_idx = 15
    output_idx = 17
    stat_mean_var = torch.load(cfg.model.kmeans.stat_mean_var_path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)

    return semantic_model, semantic_mean, semantic_std


def build_codec_model(cfg, device):
    codec_encoder = CodecEncoder(cfg=cfg.model.codec.encoder)
    codec_decoder = CodecDecoder(cfg=cfg.model.codec.decoder)
    if ".bin" in cfg.model.codec.encoder.pretrained_path:
        codec_encoder.load_state_dict(
            torch.load(cfg.model.codec.encoder.pretrained_path)
        )
        codec_decoder.load_state_dict(
            torch.load(cfg.model.codec.decoder.pretrained_path)
        )
    else:
        accelerate.load_checkpoint_and_dispatch(
            codec_encoder, cfg.model.codec.encoder.pretrained_path
        )
        accelerate.load_checkpoint_and_dispatch(
            codec_decoder, cfg.model.codec.decoder.pretrained_path
        )
    codec_encoder.eval()
    codec_decoder.eval()
    codec_encoder.to(device)
    codec_decoder.to(device)
    return codec_encoder, codec_decoder


@torch.no_grad()
def extract_acoustic_code(speech):
    vq_emb = codec_encoder(speech.unsqueeze(1))
    _, vq, _, _, _ = codec_decoder.quantizer(vq_emb)
    acoustic_code = vq.permute(
        1, 2, 0
    )  # (num_quantizer, T, C) -> (T, C, num_quantizer)
    return acoustic_code


@torch.no_grad()
def extract_semantic_code(semantic_mean, semantic_std, input_features, attention_mask):
    vq_emb = semantic_model(
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = vq_emb.hidden_states[17]  # (B, T, C)
    feat = (feat - semantic_mean.to(feat)) / semantic_std.to(feat)

    semantic_code, _ = kmeans_model.quantize(feat)  # (B, T)
    return semantic_code


@torch.no_grad()
def extract_features(speech, processor):
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"][0]
    attention_mask = inputs["attention_mask"][0]
    return input_features, attention_mask


@torch.no_grad()
def text2semantic(
    prompt0_speech,
    prompt0_text,
    prompt_speech,
    prompt_text,
    prompt_language,
    target_text,
    target_language,
    use_prompt_text=True,
    temp=1.0,
    top_k=1000,
    top_p=0.85,
    infer_mode="new",
):
    if use_prompt_text:
        if infer_mode == "new" and prompt0_speech is not None and prompt0_speech.any():
            prompt0_phone_id = new_g2p_(prompt0_text, prompt_language)[1]
            prompt0_phone_id = torch.tensor(prompt0_phone_id, dtype=torch.long).to(
                device
            )

        prompt_phone_id = new_g2p_(prompt_text, prompt_language)[1]
        prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(device)

        target_phone_id = new_g2p_(target_text, target_language)[1]
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(device)

        phone_id = torch.cat(
            [prompt_phone_id, torch.LongTensor([4]).to(device), target_phone_id]
        )

    else:
        target_phone_id = new_g2p_(target_text, target_language)[1]
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(device)
        phone_id = target_phone_id

    input_fetures, attention_mask = extract_features(prompt_speech, processor)
    input_fetures = input_fetures.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    semantic_code = extract_semantic_code(
        semantic_mean, semantic_std, input_fetures, attention_mask
    )

    if infer_mode == "new":
        input_fetures_prompt0, attention_mask_prompt0 = extract_features(
            prompt0_speech, processor
        )
        input_fetures_prompt0 = input_fetures_prompt0.unsqueeze(0).to(device)
        attention_mask_prompt0 = attention_mask_prompt0.unsqueeze(0).to(device)
        attention_mask_prompt0 = attention_mask_prompt0.float()
        semantic_code_prompt0 = extract_semantic_code(
            semantic_mean, semantic_std, input_fetures_prompt0, attention_mask_prompt0
        )

    if use_prompt_text:
        if infer_mode == "new":
            predict_semantic = t2s_model_new.sample_hf(
                phone_ids=phone_id.unsqueeze(0),
                prompt_ids=semantic_code[:, :],
                prompt0_ids=semantic_code_prompt0[:, :],
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
            )

    else:
        if infer_mode == "new":
            predict_semantic = t2s_model_new.sample_hf(
                phone_ids=phone_id.unsqueeze(0),
                prompt_ids=semantic_code[:, :1],
                prompt0_ids=semantic_code_prompt0[:, :1],
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
            )

    combine_semantic_code = torch.cat([semantic_code[:, :], predict_semantic], dim=-1)
    prompt_semantic_code = semantic_code

    return combine_semantic_code, prompt_semantic_code


@torch.no_grad()
def semantic2acoustic(combine_semantic_code, acoustic_code):

    semantic_code = combine_semantic_code

    if soundstorm_1layer.cond_code_layers == 1:
        cond = soundstorm_1layer.cond_emb(semantic_code)
    else:
        cond = soundstorm_1layer.cond_emb[0](semantic_code[0, :, :])
        for i in range(1, soundstorm_1layer.cond_code_layers):
            cond += soundstorm_1layer.cond_emb[i](semantic_code[i, :, :])
        cond = cond / math.sqrt(soundstorm_1layer.cond_code_layers)

    prompt = acoustic_code[:, :, :]
    predict_1layer = soundstorm_1layer.reverse_diffusion(
        cond=cond,
        prompt=prompt,
        temp=1.5,
        filter_thres=0.98,
        n_timesteps=[40],
        cfg=1.0,
        rescale_cfg=1.0,
    )

    if soundstorm_full.cond_code_layers == 1:
        cond = soundstorm_full.cond_emb(semantic_code)
    else:
        cond = soundstorm_full.cond_emb[0](semantic_code[0, :, :])
        for i in range(1, soundstorm_full.cond_code_layers):
            cond += soundstorm_full.cond_emb[i](semantic_code[i, :, :])
        cond = cond / math.sqrt(soundstorm_full.cond_code_layers)

    prompt = acoustic_code[:, :, :]
    predict_full = soundstorm_full.reverse_diffusion(
        cond=cond,
        prompt=prompt,
        temp=1.5,
        filter_thres=0.98,
        n_timesteps=[40, 16, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        cfg=1.0,
        rescale_cfg=1.0,
        gt_code=predict_1layer,
    )
    vq_emb = codec_decoder.vq2emb(predict_full.permute(2, 0, 1), n_quantizers=12)
    recovered_audio = codec_decoder(vq_emb)
    prompt_vq_emb = codec_decoder.vq2emb(prompt.permute(2, 0, 1), n_quantizers=12)
    recovered_prompt_audio = codec_decoder(prompt_vq_emb)
    recovered_prompt_audio = recovered_prompt_audio[0][0].cpu().numpy()
    recovered_audio = recovered_audio[0][0].cpu().numpy()
    combine_audio = np.concatenate([recovered_prompt_audio, recovered_audio])

    return combine_audio, recovered_audio


device = torch.device("cuda:0")
cfg_soundstorm_1layer = load_config("./s2a_egs/s2a_debatts_1layer.json")
cfg_soundstorm_full = load_config("./s2a_egs/s2a_debatts_full.json")

soundstorm_1layer = build_soundstorm(cfg_soundstorm_1layer, device)
soundstorm_full = build_soundstorm(cfg_soundstorm_full, device)

semantic_model, semantic_mean, semantic_std = build_semantic_model(
    cfg_soundstorm_full, device
)
kmeans_model = build_kmeans_model(cfg_soundstorm_full, device)

codec_encoder, codec_decoder = build_codec_model(cfg_soundstorm_full, device)

semantic_model, semantic_mean, semantic_std = build_semantic_model(
    cfg_soundstorm_full, device
)
kmeans_model = build_kmeans_model(cfg_soundstorm_full, device)

soundstorm_1layer_path = "./s2a_model/s2a_model_1layer/onelayer_model.safetensors"
soundstorm_full_path = "./s2a_model/s2a_model_full/full_model.safetensors"
safetensors.torch.load_model(soundstorm_1layer, soundstorm_1layer_path)
safetensors.torch.load_model(soundstorm_full, soundstorm_full_path)

t2s_cfg = load_config("./t2s_egs/t2s_debatts.json")
t2s_model_new = build_t2s_model_new(t2s_cfg, device)
t2s_model_new_ckpt_path = "./t2s_model/model.safetensors"
safetensors.torch.load_model(t2s_model_new, t2s_model_new_ckpt_path)

from funasr import AutoModel

print("Loading ASR model...")
asr_model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model="ct-punc",
    device="cuda:0",
)


def adjust_punctuation(text):
    """
    Adjust the punctuation so that the comma is followed
    by a space and the rest of the punctuation uses the
    full Angle symbol.
    """
    text = text.replace("，", ", ")

    punct_mapping = {
        "。": "。",
        "？": "？",
        "！": "！",
        "：": "：",
        "；": "；",
        "“": "“",
        "”": "”",
        "‘": "‘",
        "’": "’",
    }
    for punct, full_punct in punct_mapping.items():
        text = text.replace(punct, full_punct)
    return text


import random
import zhconv


def generate_text_data(wav_file):
    idx = random.randint(0, 7000)
    speech = librosa.load(wav_file, sr=16000)[0]
    txt_json_path = wav_file.replace(".wav", ".json")
    txt_json_param_path = wav_file.replace(".wav", "_asr_param.json")
    if os.path.exists(txt_json_path):

        with open(txt_json_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        if "text" in json_data:
            txt = json_data["text"]
            txt = adjust_punctuation(txt)

    elif os.path.exists(txt_json_param_path):
        with open(txt_json_param_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        if "text" in json_data:
            txt = json_data["text"]
            txt = adjust_punctuation(txt)

        else:
            res = asr_model.generate(input=wav_file, batch_size_s=300)
            txt = res[0]["text"]
            txt = zhconv.convert(txt, "zh-cn")
            txt = adjust_punctuation(txt)

            json_data["text"] = txt
            with open(txt_json_path, "w", encoding="utf-8") as file:
                json.dump(json_data, file, ensure_ascii=False, indent=4)

    # If no JSON file is found, generate new text and save it to a new JSON file
    else:
        res = asr_model.generate(input=wav_file, batch_size_s=300)
        txt = res[0]["text"]
        txt = zhconv.convert(txt, "zh-cn")
        txt = adjust_punctuation(txt)
        # txt = re.sub(" ", "", txt)

        json_data = {"text": txt}
        with open(txt_json_path, "w", encoding="utf-8") as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)

    return wav_file, txt, wav_file


def infer(
    speech_path,
    prompt_text,
    target_wav_path,
    target_text,
    target_language="zh",
    speech_path_prompt0=None,
    prompt0_text=None,
    temperature=0.2,
    top_k=20,
    top_p=0.9,
    concat_prompt=False,
    infer_mode="new",
    idx=0,
    epoch=0,
    spk_prompt_type="",
):
    if idx != 0:
        save_dir = os.path.join(
            "The Path to Store Generated Speech", f"{infer_mode}/{spk_prompt_type}"
        )
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(
            save_dir,
            f"{os.path.splitext(os.path.basename(target_wav_path))[0]}_infer_{infer_mode}_{idx}_epoch_{epoch}_{spk_prompt_type}.wav",
        )
    else:
        save_dir = os.path.join(
            "The Path to Store Generated Speech", f"{infer_mode}/{spk_prompt_type}"
        )
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(
            save_dir,
            f"{os.path.splitext(os.path.basename(target_wav_path))[0]}_infer_{infer_mode}_epoch_{epoch}_{spk_prompt_type}.wav",
        )

    if os.path.exists(save_path):
        return save_path

    # print(f"HERE COMES INFER!!! {infer_mode}")
    # print(f"IN INFER PROMPT text is {prompt_text}")
    # print(f"IN INFER Target text is {target_text}")
    speech_16k = librosa.load(speech_path, sr=16000)[0]
    speech = librosa.load(speech_path, sr=cfg_soundstorm_1layer.preprocess.sample_rate)[
        0
    ]

    if infer_mode == "new":
        speech_16k_prompt0 = librosa.load(speech_path_prompt0, sr=16000)[0]
        speech_prompt0 = librosa.load(
            speech_path_prompt0, sr=cfg_soundstorm_1layer.preprocess.sample_rate
        )[0]
        combine_semantic_code, _ = text2semantic(
            prompt0_speech=speech_16k_prompt0,
            prompt0_text=prompt0_text,
            prompt_speech=speech_16k,
            prompt_text=prompt_text,
            prompt_language=target_language,
            target_text=target_text,
            target_language=target_language,
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            infer_mode=infer_mode,
        )

    else:
        combine_semantic_code, _ = text2semantic(
            prompt0_speech=None,
            prompt0_text=None,
            prompt_speech=speech_16k,
            prompt_text=prompt_text,
            prompt_language=target_language,
            target_text=target_text,
            target_language=target_language,
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            infer_mode=infer_mode,
        )
    acoustic_code = extract_acoustic_code(torch.tensor(speech).unsqueeze(0).to(device))
    combine_audio, recovered_audio = semantic2acoustic(
        combine_semantic_code, acoustic_code
    )

    if not concat_prompt:
        combine_audio = combine_audio[speech.shape[-1] :]
    # sf.write(os.path.join(save_path, "{}.wav".format(uid)), recovered_audio, samplerate=cfg_soundstorm_1layer.preprocess.sample_rate)
    sf.write(
        save_path,
        combine_audio,
        samplerate=cfg_soundstorm_1layer.preprocess.sample_rate,
    )
    return save_path


def infer_small(
    speech_path,
    prompt_text,
    target_text,
    target_language="zh",
    speech_path_prompt0=None,
    prompt0_text=None,
    temperature=0.2,
    top_k=20,
    top_p=0.9,
    concat_prompt=False,
    infer_mode="new",
    save_path=None,
):

    if os.path.exists(save_path):
        return save_path

    speech_16k = librosa.load(speech_path, sr=16000)[0]
    speech = librosa.load(speech_path, sr=cfg_soundstorm_1layer.preprocess.sample_rate)[
        0
    ]

    if infer_mode == "new":
        speech_16k_prompt0 = librosa.load(speech_path_prompt0, sr=16000)[0]
        speech_prompt0 = librosa.load(
            speech_path_prompt0, sr=cfg_soundstorm_1layer.preprocess.sample_rate
        )[0]
        # combine_semantic_code, _ = text2semantic_new(speech_16k_prompt0, prompt0_text, speech_16k, prompt_text, target_language, target_text, target_language, temp=temperature, top_k=top_k, top_p=top_p, infer_mode=infer_mode)
        combine_semantic_code, _ = text2semantic(
            prompt0_speech=speech_16k_prompt0,
            prompt0_text=prompt0_text,
            prompt_speech=speech_16k,
            prompt_text=prompt_text,
            prompt_language=target_language,
            target_text=target_text,
            target_language=target_language,
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            infer_mode=infer_mode,
        )

    else:
        combine_semantic_code, _ = text2semantic(
            prompt0_speech=None,
            prompt0_text=None,
            prompt_speech=speech_16k,
            prompt_text=prompt_text,
            prompt_language=target_language,
            target_text=target_text,
            target_language=target_language,
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            infer_mode=infer_mode,
        )
    acoustic_code = extract_acoustic_code(torch.tensor(speech).unsqueeze(0).to(device))
    combine_audio, recovered_audio = semantic2acoustic(
        combine_semantic_code, acoustic_code
    )

    if not concat_prompt:
        combine_audio = combine_audio[speech.shape[-1] :]
    # sf.write(os.path.join(save_path, "{}.wav".format(uid)), recovered_audio, samplerate=cfg_soundstorm_1layer.preprocess.sample_rate)
    sf.write(
        save_path,
        combine_audio,
        samplerate=cfg_soundstorm_1layer.preprocess.sample_rate,
    )
    return save_path


##################################### EVALUATION ################################################################
from funasr import AutoModel
import torch.nn.functional as F
import torch

from models.tts.soundstorm.try_inference_new import evaluation
from models.tts.soundstorm.try_inference_new import evaluation_new
from models.tts.soundstorm.try_inference_new import extract_emotion_similarity

prompt0_wav_path = "./speech_examples/87_SPEAKER01_2_part03_213.wav"
prompt0_text = generate_text_data(prompt0_wav_path)[1]

spk_prompt_wav_path = "./speech_examples/87_SPEAKER00_7_part11_212_prompt.wav"
spk_prompt_text = generate_text_data(spk_prompt_wav_path)[1]

# TODO
save_path_dir = "The Path to Save Generated Speech"
wav_filename = "The Filename of Generated Speech"
save_path = os.path.join(save_path_infer_dir, wav_filename)
save_path = infer_small(
    speech_path=spk_prompt_wav_path,
    prompt_text=spk_prompt_text,
    target_text=target_text,
    speech_path_prompt0=prompt0_wav_path,
    prompt0_text=prompt0_text,
    infer_mode="new",
    save_path=save_path,
)
