# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import spaces
import accelerate
import gradio as gr
import torch
import safetensors
from huggingface_hub import hf_hub_download
import soundfile as sf
import os

import numpy as np
import librosa
from models.codec.kmeans.repcodec_model import RepCodec
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from transformers import Wav2Vec2BertModel
from utils.util import load_config
from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p

from transformers import SeamlessM4TFeatureExtractor
import py3langid as langid
import whisper

print("Start loading: facebook/w2v-bert-2.0")
processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
whisper_model = None
output_file_name_idx = 0


def detect_text_language(text):
    return langid.classify(text)[0]


def detect_speech_language(speech_file):
    global whisper_model
    if whisper_model == None:
        whisper_model = whisper.load_model("turbo")
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(speech_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(whisper_model.device)

    # detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    return max(probs, key=probs.get)


@torch.no_grad()
def get_prompt_text(speech_16k, language):
    full_prompt_text = ""
    shot_prompt_text = ""
    short_prompt_end_ts = 0.0

    global whisper_model
    if whisper_model == None:
        whisper_model = whisper.load_model("turbo")
    asr_result = whisper_model.transcribe(speech_16k, language=language)
    full_prompt_text = asr_result["text"]  # whisper asr result
    # text = asr_result["segments"][0]["text"] # whisperx asr result
    shot_prompt_text = ""
    short_prompt_end_ts = 0.0
    for segment in asr_result["segments"]:
        shot_prompt_text = shot_prompt_text + segment["text"]
        short_prompt_end_ts = segment["end"]
        if short_prompt_end_ts >= 4:
            break
    return full_prompt_text, shot_prompt_text, short_prompt_end_ts


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


@torch.no_grad()
def extract_features(speech, processor):
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"][0]
    attention_mask = inputs["attention_mask"][0]
    return input_features, attention_mask


@torch.no_grad()
def extract_semantic_code(semantic_mean, semantic_std, input_features, attention_mask):
    vq_emb = semantic_model(
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = vq_emb.hidden_states[17]  # (B, T, C)
    feat = (feat - semantic_mean.to(feat)) / semantic_std.to(feat)

    semantic_code, rec_feat = semantic_codec.quantize(feat)  # (B, T)
    return semantic_code, rec_feat


@torch.no_grad()
def extract_acoustic_code(speech):
    vq_emb = codec_encoder(speech.unsqueeze(1))
    _, vq, _, _, _ = codec_decoder.quantizer(vq_emb)
    acoustic_code = vq.permute(1, 2, 0)
    return acoustic_code


@torch.no_grad()
def text2semantic(
    device,
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

    if target_len < 0:
        target_len = int(
            (len(prompt_speech) * len(target_phone_id) / len(prompt_phone_id))
            / 16000
            * 50
        )
    else:
        target_len = int(target_len * 50)

    prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(device)
    target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(device)

    phone_id = torch.cat([prompt_phone_id, target_phone_id])

    input_fetures, attention_mask = extract_features(prompt_speech, processor)
    input_fetures = input_fetures.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    semantic_code, _ = extract_semantic_code(
        semantic_mean, semantic_std, input_fetures, attention_mask
    )

    predict_semantic = t2s_model.reverse_diffusion(
        semantic_code[:, :],
        target_len,
        phone_id.unsqueeze(0),
        n_timesteps=n_timesteps,
        cfg=cfg,
        rescale_cfg=rescale_cfg,
    )

    combine_semantic_code = torch.cat([semantic_code[:, :], predict_semantic], dim=-1)
    prompt_semantic_code = semantic_code

    return combine_semantic_code, prompt_semantic_code


@torch.no_grad()
def semantic2acoustic(
    device,
    combine_semantic_code,
    acoustic_code,
    n_timesteps=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    cfg=2.5,
    rescale_cfg=0.75,
):

    semantic_code = combine_semantic_code

    cond = s2a_model_1layer.cond_emb(semantic_code)
    prompt = acoustic_code[:, :, :]
    predict_1layer = s2a_model_1layer.reverse_diffusion(
        cond=cond,
        prompt=prompt,
        temp=1.5,
        filter_thres=0.98,
        n_timesteps=n_timesteps[:1],
        cfg=cfg,
        rescale_cfg=rescale_cfg,
    )

    cond = s2a_model_full.cond_emb(semantic_code)
    prompt = acoustic_code[:, :, :]
    predict_full = s2a_model_full.reverse_diffusion(
        cond=cond,
        prompt=prompt,
        temp=1.5,
        filter_thres=0.98,
        n_timesteps=n_timesteps,
        cfg=cfg,
        rescale_cfg=rescale_cfg,
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


# Load the model and checkpoints
def load_models():
    cfg_path = "./models/tts/maskgct/config/maskgct.json"

    cfg = load_config(cfg_path)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    codec_encoder, codec_decoder = build_acoustic_codec(
        cfg.model.acoustic_codec, device
    )
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)
    print("Models built successfully.")

    # Download checkpoints
    semantic_code_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
    )
    codec_encoder_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="acoustic_codec/model.safetensors"
    )
    codec_decoder_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors"
    )
    t2s_model_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="t2s_model/model.safetensors"
    )
    s2a_1layer_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors"
    )
    s2a_full_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors"
    )
    print("Checkpoints downloaded successfully.")

    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    # accelerate.load_checkpoint_and_dispatch(codec_encoder, codec_encoder_ckpt)
    # accelerate.load_checkpoint_and_dispatch(codec_decoder, codec_decoder_ckpt)
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)
    print("Checkpoints loaded successfully.")

    return (
        semantic_model,
        semantic_mean,
        semantic_std,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
    )


@torch.no_grad()
def maskgct_inference(
    prompt_speech_path,
    target_text,
    target_len=None,
    n_timesteps=25,
    cfg=2.5,
    rescale_cfg=0.75,
    n_timesteps_s2a=[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    cfg_s2a=2.5,
    rescale_cfg_s2a=0.75,
    device=torch.device("cuda:0"),
):
    speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]
    speech = librosa.load(prompt_speech_path, sr=24000)[0]
    print("Audio loaded.")

    prompt_language = detect_speech_language(prompt_speech_path)
    full_prompt_text, short_prompt_text, shot_prompt_end_ts = get_prompt_text(
        prompt_speech_path, prompt_language
    )
    # use the first 4+ seconds wav as the prompt in case the prompt wav is too long
    speech = speech[0 : int(shot_prompt_end_ts * 24000)]
    speech_16k = speech_16k[0 : int(shot_prompt_end_ts * 16000)]

    target_language = detect_text_language(target_text)
    combine_semantic_code, _ = text2semantic(
        device,
        speech_16k,
        short_prompt_text,
        prompt_language,
        target_text,
        target_language,
        target_len,
        n_timesteps,
        cfg,
        rescale_cfg,
    )
    acoustic_code = extract_acoustic_code(torch.tensor(speech).unsqueeze(0).to(device))
    _, recovered_audio = semantic2acoustic(
        device,
        combine_semantic_code,
        acoustic_code,
        n_timesteps=n_timesteps_s2a,
        cfg=cfg_s2a,
        rescale_cfg=rescale_cfg_s2a,
    )

    return recovered_audio


@spaces.GPU
def inference(
    prompt_wav,
    target_text,
    target_len,
    n_timesteps,
):
    print(f"===== New task submitted =====")
    global output_file_name_idx
    save_path = f"./output/output_{output_file_name_idx}.wav"
    os.makedirs("./output", exist_ok=True)
    print(f"Start inference...")
    recovered_audio = maskgct_inference(
        prompt_wav,
        target_text,
        target_len=target_len,
        n_timesteps=int(n_timesteps),
        device=device,
    )
    print(f"Saved: {save_path}")
    sf.write(save_path, recovered_audio, 24000)
    output_file_name_idx = (output_file_name_idx + 1) % 100
    return save_path


# Load models once
(
    semantic_model,
    semantic_mean,
    semantic_std,
    semantic_codec,
    codec_encoder,
    codec_decoder,
    t2s_model,
    s2a_model_1layer,
    s2a_model_full,
) = load_models()

# Language list
language_list = ["en", "zh", "ja", "ko", "fr", "de"]

# Gradio interface
iface = gr.Interface(
    fn=inference,
    inputs=[
        gr.Audio(label="Upload Prompt Wav", type="filepath"),
        gr.Textbox(label="Target Text"),
        gr.Number(
            label="Target Duration (in seconds), if the target duration is less than 0, the system will estimate a duration.",
            value=-1,
        ),  # Removed 'optional=True'
        gr.Slider(
            label="Number of Timesteps", minimum=15, maximum=100, value=25, step=1
        ),
    ],
    outputs=gr.Audio(label="Generated Audio"),
    title="MaskGCT TTS Demo",
    description="""
    [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2409.00750) [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/maskgct) [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-demo-pink)](https://huggingface.co/spaces/amphion/maskgct) [![readme](https://img.shields.io/badge/README-Key%20Features-blue)](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
    """,
)

# Launch the interface
iface.launch(allowed_paths=["./output"])
