# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import json
import librosa
import torch
import torchaudio
import accelerate
import safetensors
import numpy as np
import os
import yaml
from IPython.display import display, Audio

import parselmouth
import torchvision
import random
import numpy as np
import whisper
from librosa.feature import chroma_stft
from librosa.effects import pitch_shift

from models.codec.coco.rep_coco_model import CocoContentStyle, CocoContent, CocoStyle
from models.svc.flow_matching_transformer.fmt_model import FlowMatchingTransformer
from models.svc.autoregressive_transformer.ar_model import AutoregressiveTransformer
from models.codec.melvqgan.melspec import MelSpectrogram
from models.codec.amphion_codec.vocos import Vocos

from utils.util import load_config
from evaluation.metrics.f0.f0_corr import extract_f0_hz


def g2p_(text, language):
    from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p

    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return g2p(text, sentence=None, language=language)


# Coco Tokenizer
def build_coco_model(coco_cfg, device):
    coco_model_type = getattr(coco_cfg, "coco_type", "content_style")
    if coco_model_type == "content_style":
        model = CocoContentStyle(cfg=coco_cfg, construct_only_for_quantizer=True)
    elif coco_model_type == "content":
        model = CocoContent(cfg=coco_cfg, construct_only_for_quantizer=True)
    elif coco_model_type == "style":
        model = CocoStyle(cfg=coco_cfg, construct_only_for_quantizer=True)
    else:
        raise ValueError(f"Unknown coco type: {coco_model_type}")

    model.eval()
    model.to(device)
    return model


# Flow Matching Transformer
def build_fmt_model(cfg, device):
    model = FlowMatchingTransformer(cfg=cfg.model.flow_matching_transformer)
    model.eval()
    model.to(device)
    return model


# Autoregressive Transformer
def build_ar_model(cfg, device):
    model = AutoregressiveTransformer(cfg=cfg.model.autoregressive_transformer)
    model.eval()
    model.to(device)
    return model


# Melspectrogram Extractor
def build_mel_model(cfg, device):
    mel_model = MelSpectrogram(
        sampling_rate=cfg.preprocess.sample_rate,
        n_fft=cfg.preprocess.n_fft,
        num_mels=cfg.preprocess.num_mels,
        hop_size=cfg.preprocess.hop_size,
        win_size=cfg.preprocess.win_size,
        fmin=cfg.preprocess.fmin,
        fmax=cfg.preprocess.fmax,
    )
    mel_model.eval()
    mel_model.to(device)
    return mel_model


# Vocoder
def build_vocoder_model(cfg, device):
    vocoder_model = Vocos(cfg=cfg.model.vocos)
    vocoder_model.eval()
    vocoder_model.to(device)
    return vocoder_model


def load_checkpoint(build_model_func, cfg, ckpt_path, device):
    model = build_model_func(cfg, device)
    accelerate.load_checkpoint_and_dispatch(model, ckpt_path)
    return model


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1e6:
        return f"{total_params} params"  # Parameters
    elif total_params < 1e9:
        return f"{total_params / 1e6:.2f} M"  # Millions
    else:
        return f"{total_params / 1e9:.2f} B"  # Billions


def load_wav(wav_path, device):
    speech = librosa.load(wav_path, sr=24000)[0]
    speech_tensor = torch.tensor(speech).unsqueeze(0).to(device)
    speech16k = torchaudio.functional.resample(speech_tensor, 24000, 16000)
    return speech, speech_tensor, speech16k


def display_audio_in_notebook(wav, rate=24000):
    display(Audio(wav, rate=rate))


def save_audio(
    waveform, sr=24000, output_path=None, target_sample_rate=None, target_db=-25.0
):
    """
    waveform: [1, T]
    """
    if target_sample_rate is not None and sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
    else:
        target_sample_rate = sr

    rms = torch.sqrt(torch.mean(waveform**2))
    current_db = 20 * torch.log10(rms + 1e-9)

    gain = target_db - current_db
    normalized_waveform = waveform * (10 ** (gain / 20))

    torchaudio.save(output_path, normalized_waveform, target_sample_rate)
    return output_path


class VevosingInferencePipeline:
    def __init__(
        self,
        prosody_tokenizer_ckpt_path=None,
        content_style_tokenizer_ckpt_path=None,
        ar_cfg_path=None,
        ar_ckpt_path=None,
        fmt_cfg_path=None,
        fmt_ckpt_path=None,
        vocoder_cfg_path=None,
        vocoder_ckpt_path=None,
        device=None,
    ):
        self.device = device
        self.prosody_tokenizer_ckpt_path = prosody_tokenizer_ckpt_path
        self.content_style_tokenizer_ckpt_path = content_style_tokenizer_ckpt_path

        if ar_cfg_path is not None and ar_ckpt_path is not None:
            self.ar_cfg = load_config(ar_cfg_path)
            self.ar_model = load_checkpoint(
                build_ar_model, self.ar_cfg, ar_ckpt_path, device
            )
            print(f"#Params of AR model: {count_parameters(self.ar_model)}")
        else:
            self.ar_cfg = None
            self.ar_model = None

        if fmt_cfg_path is not None and fmt_ckpt_path is not None:
            self.fmt_cfg = load_config(fmt_cfg_path)
            self.fmt_model = load_checkpoint(
                build_fmt_model, self.fmt_cfg, fmt_ckpt_path, device
            )
            print(f"#Params of Flow Matching model: {count_parameters(self.fmt_model)}")

            self.init_coco_tokenizer()

        if vocoder_cfg_path is not None and vocoder_ckpt_path is not None:
            self.vocoder_cfg = load_config(vocoder_cfg_path)
            self.mel_model = build_mel_model(self.vocoder_cfg, device)
            self.vocoder_model = load_checkpoint(
                build_vocoder_model, self.vocoder_cfg, vocoder_ckpt_path, device
            )
            print(f"#Params of Vocoder model: {count_parameters(self.vocoder_model)}")

    def init_coco_tokenizer(self):
        ## Whisper ##
        self.whisper_model = whisper.load_model("medium", self.device)  # 1024 dim
        self.whisper_model.eval()

        self.use_normed_whisper = getattr(
            self.fmt_cfg.model.coco, "use_normed_whisper", False
        )
        if self.use_normed_whisper:
            whisper_stats = torch.load(
                self.fmt_cfg.model.coco.whisper_stats_path,
                map_location=self.device,
            )
            self.whisper_mean = whisper_stats["mean"]  # (1024,)
            self.whisper_std = whisper_stats["std"]  # (1024,)

        ## Content Tokenizer and Style Tokenizer ##
        if self.ar_model is not None:
            if self.ar_cfg.model.use_style_tokens_as_input:
                self.style_tokenizer = load_checkpoint(
                    build_coco_model,
                    self.ar_cfg.model.coco_style,
                    self.prosody_tokenizer_ckpt_path,
                    self.device,
                )
                print(
                    f"#Params of CocoStyle model: {count_parameters(self.style_tokenizer)}"
                )

        ## Content-Style Tokenizer ##
        self.content_style_tokenizer = load_checkpoint(
            build_coco_model,
            self.fmt_cfg.model.coco,
            self.content_style_tokenizer_ckpt_path,
            self.device,
        )
        print(
            f"#Params of CocoContentStyle model: {count_parameters(self.content_style_tokenizer)}"
        )

    @torch.no_grad()
    def extract_mel_feature(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.vocoder_cfg.preprocess.mel_mean) / math.sqrt(
            self.vocoder_cfg.preprocess.mel_var
        )
        return mel_feature

    def spec_augment(self, mel, height):
        """
        Args:
            mel: tensor (..., n_mels, frames)
            height: int 68-92 for default 80 mels
        """
        tgt = torchvision.transforms.functional.resize(mel, (height, mel.shape[-1]))
        if height >= mel.shape[-2]:
            return tgt[:, : mel.shape[-2], :]
        else:
            silence = tgt[:, -1:, :].repeat(1, mel.shape[-2] - height, 1)
            silence += torch.randn_like(silence) / 10
            return torch.cat((tgt, silence), 1)

    @torch.no_grad()
    def extract_whisper_features(self, wavs, frame_lens, spec_perturb=False):
        """
        Args:
            wavs: (B, T) at 16khz. Note that the max duration should be 30s
            frame_lens: (B,)
        Returns:
            features: (B, T, D)
        """
        # wavs: (batch, max_len)
        wavs = whisper.pad_or_trim(wavs)
        # batch_mel: (batch, 80, 3000)
        batch_mel = whisper.log_mel_spectrogram(wavs, device=self.device)

        if spec_perturb:
            height = random.randint(68, 92)
            batch_mel = self.spec_augment(batch_mel, height)

        with torch.no_grad():
            # (batch, 1500, 1024)
            features = self.whisper_model.embed_audio(batch_mel)

        max_len = int(frame_lens.max().item())
        mask = torch.arange(features.size(1), device=features.device).expand(
            len(frame_lens), -1
        ) < frame_lens.unsqueeze(1)
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))

        if features.shape[1] >= max_len:
            features = features[:, :max_len, :]
        else:
            padding_frames = max_len - features.shape[1]
            last_frame = features[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            features = torch.cat([features, padding], dim=1)

        if self.use_normed_whisper:
            features = (features - self.whisper_mean) / self.whisper_std

        return features

    @torch.no_grad()
    def extract_coco_codec(
        self,
        coco_codec_type,
        wav16k,
        wav24k_numpy,
        whisper_spec_perturb=False,
        frame_len_ratio=1.0,
        use_shifted_src_to_extract_prosody=False,
        use_shifted_src_to_extract_contentstyle=False,
        src_shifted_steps=0,
    ):
        """
        Args:
            coco_codec_type: "content", "style", or "content_style"
            wav16k: [1, T]
            wav24k_numpy: [T]
        Returns:
            codecs: [1, T]. Note that codecs might be not at 50Hz!
        """
        frame_len = len(wav24k_numpy) // self.fmt_cfg.preprocess.hop_size

        if not use_shifted_src_to_extract_prosody:
            chromagram_feats = self.get_chromagram(wav24k_numpy, frame_len)  # [T, 24]
        else:
            chromagram_feats = self.get_chromagram(
                pitch_shift(wav24k_numpy, sr=24000, n_steps=src_shifted_steps),
                frame_len,
            )  # [T, 24]

        chromagram_feats = (
            torch.tensor(chromagram_feats, dtype=torch.float)
            .unsqueeze(0)
            .to(self.device)
        )  # [1, T, 24]

        if frame_len_ratio != 1.0:
            raw_len = chromagram_feats.shape[1]
            # 将 [1, T, 24] 转换为 [1, 24, T] 以便在最后一个维度上进行插值
            chromagram_feats = chromagram_feats.transpose(1, 2)
            chromagram_feats = torch.nn.functional.interpolate(
                chromagram_feats,
                size=int(raw_len * frame_len_ratio),  # 明确指定目标长度
                mode="linear",
                align_corners=False,
            )  # [1, 24, T']
            # 转换回原始形状 [1, T', 24]
            chromagram_feats = chromagram_feats.transpose(1, 2)
            print(
                f"Chromagram feats are sampled from {raw_len} to {chromagram_feats.shape[1]}, ratio = {frame_len_ratio}"
            )

        if use_shifted_src_to_extract_contentstyle:
            wav16k_shifted = pitch_shift(
                wav16k.cpu().numpy()[0], sr=16000, n_steps=src_shifted_steps
            )  # [T]
            wav16k = torch.tensor(wav16k_shifted).unsqueeze(0).to(self.device)  # [1, T]

        whisper_feats = self.extract_whisper_features(
            wav16k,
            torch.tensor([frame_len], dtype=torch.long).to(self.device),
            spec_perturb=whisper_spec_perturb,
        )  # [1, T, D]

        if coco_codec_type == "content_style":
            codecs, _ = self.content_style_tokenizer.quantize(
                whisper_feats.to(torch.float32), chromagram_feats.to(torch.float32)
            )
        elif coco_codec_type == "content":
            codecs, _ = self.content_tokenizer.quantize(whisper_feats.to(torch.float32))
        elif coco_codec_type == "style":
            codecs, _ = self.style_tokenizer.quantize(
                chromagram_feats.to(torch.float32)
            )
        else:
            raise ValueError(f"Unknown coco type: {coco_codec_type}")

        return codecs

    def get_chromagram(self, speech, speech_frames):
        # [24, T] -> [T, 24]
        chromagram = chroma_stft(
            y=speech,
            sr=self.fmt_cfg.preprocess.sample_rate,
            n_fft=self.fmt_cfg.preprocess.n_fft,
            hop_length=self.fmt_cfg.preprocess.hop_size,
            win_length=self.fmt_cfg.preprocess.win_size,
            n_chroma=24,
        ).T

        if chromagram.shape[0] < speech_frames:
            chromagram = np.pad(
                chromagram, (0, speech_frames - chromagram.shape[0]), mode="edge"
            )
        else:
            chromagram = chromagram[:speech_frames]

        return chromagram

    def inference_fm(
        self,
        src_wav_path,
        timbre_ref_wav_path,
        whisper_spec_perturb=False,
        use_shifted_src_to_extract_prosody=False,
        use_shifted_src_to_extract_contentstyle=False,
        flow_matching_steps=32,
        display_audio=False,
    ):
        src_speech, src_speech24k, src_speech16k = load_wav(src_wav_path, self.device)
        timbre_ref_speech, timbre_ref_speech24k, timbre_ref_speech16k = load_wav(
            timbre_ref_wav_path, self.device
        )

        if display_audio:
            print("-" * 20)
            if src_wav_path == timbre_ref_wav_path:
                print("We want to reconstruct this audio:", src_wav_path)
                display_audio_in_notebook(src_wav_path, rate=24000)
            else:
                print("Source audio:")
                display_audio_in_notebook(src_speech, rate=24000)

        ## Whether to use shifted src to extract prosody and content-style ##
        if (
            use_shifted_src_to_extract_prosody
            or use_shifted_src_to_extract_contentstyle
        ):
            src_f0 = extract_f0_hz(src_wav_path)
            timbre_ref_f0 = extract_f0_hz(timbre_ref_wav_path)

            src_f0_median = np.median(src_f0)
            timbre_ref_f0_median = np.median(timbre_ref_f0)

            src_shifted_steps = 12 * np.log2(timbre_ref_f0_median / src_f0_median)
            src_shifted_steps = round(src_shifted_steps)

            if display_audio:
                print("-" * 20)
                print(
                    f"src_wav f0 median: {src_f0_median} hz, timbre_ref_wav f0 median: {timbre_ref_f0_median} hz, src_shifted_steps: {src_shifted_steps}"
                )
        else:
            src_shifted_steps = 0

        ## Diffusion ##
        src_codecs = self.extract_coco_codec(
            "content_style",
            src_speech16k,
            src_speech,
            whisper_spec_perturb=whisper_spec_perturb,
            use_shifted_src_to_extract_prosody=use_shifted_src_to_extract_prosody,
            use_shifted_src_to_extract_contentstyle=use_shifted_src_to_extract_contentstyle,
            src_shifted_steps=src_shifted_steps,
        )  # [1, T]
        timbre_ref_codecs = self.extract_coco_codec(
            "content_style",
            timbre_ref_speech16k,
            timbre_ref_speech,
            whisper_spec_perturb=whisper_spec_perturb,
        )  # [1, T]
        diffusion_input_codecs = torch.cat([timbre_ref_codecs, src_codecs], dim=1)

        # Prepare the condition for diffusion
        diffusion_cond = self.fmt_model.cond_emb(diffusion_input_codecs)  # [1, T, D]
        if self.fmt_model.do_resampling:
            # Align to the frame rate of Mels
            diffusion_cond = self.fmt_model.resampling_layers(
                diffusion_cond.transpose(1, 2)
            ).transpose(1, 2)

        src_mels = self.extract_mel_feature(src_speech24k)  # [1, T, D]
        timbre_ref_mels = self.extract_mel_feature(timbre_ref_speech24k)  # [1, T, D]
        T = timbre_ref_mels.shape[1] + src_mels.shape[1]
        if diffusion_cond.shape[1] >= T:  # Check time dimension
            diffusion_cond = diffusion_cond[:, :T, :]
        else:
            padding_frames = T - diffusion_cond.shape[1]
            last_frame = diffusion_cond[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            diffusion_cond = torch.cat([diffusion_cond, padding], dim=1)

        # [1, T, D]
        predict_mel_feat = self.fmt_model.reverse_diffusion(
            cond=diffusion_cond,
            prompt=timbre_ref_mels,
            n_timesteps=flow_matching_steps,
        )

        ## Vocoder and Display ##
        # [1, 1, T] -> [1, T]
        synthesized_audio = (
            self.vocoder_model(predict_mel_feat.transpose(1, 2)).detach().cpu()
        )[0]
        if display_audio:
            # [T]
            audio = synthesized_audio.numpy()[0]
            display_audio_in_notebook(audio, rate=24000)

        return synthesized_audio

    def inference_ar_and_fm(
        self,
        task="conversion",  # "synthesis", "conversion", "recognition-synthesis"
        src_wav_path=None,
        src_text=None,
        src_text_language=None,
        style_ref_wav_path=None,
        style_ref_wav_text=None,
        style_ref_wav_text_language=None,
        timbre_ref_wav_path=None,
        use_style_tokens_as_ar_input=False,
        target_src_duration_ratio=1.0,
        flow_matching_steps=32,
        display_audio=False,
    ):
        assert self.ar_model is not None

        if src_wav_path is not None:
            src_speech, src_speech24k, src_speech16k = load_wav(
                src_wav_path, self.device
            )
            if display_audio:
                print("-" * 20)
                print("Source audio:")
                display_audio_in_notebook(src_speech, rate=24000)

        if src_text is not None:
            if src_text_language is None:
                src_text_language = "zh"

            if display_audio:
                print("-" * 20)
                print("Source Text: [{}]".format(src_text))

        if style_ref_wav_path is not None:
            style_ref_speech, style_ref_speech24k, style_ref_speech16k = load_wav(
                style_ref_wav_path, self.device
            )

        assert timbre_ref_wav_path is not None
        timbre_ref_speech, timbre_ref_speech24k, timbre_ref_speech16k = load_wav(
            timbre_ref_wav_path, self.device
        )

        if display_audio:
            if style_ref_wav_path == timbre_ref_wav_path:
                print("Both Style and Timbre Reference audio:")
                display_audio_in_notebook(style_ref_speech, rate=24000)
            else:
                if style_ref_wav_path is not None:
                    print("Style Reference audio:")
                    display_audio_in_notebook(style_ref_speech, rate=24000)
                print("Timbre Reference audio:")
                display_audio_in_notebook(timbre_ref_speech, rate=24000)
                print("-" * 20)

        ## AR ##
        if task in ["synthesis", "recognition-synthesis"]:
            ar_input_content_ids = g2p_(src_text, src_text_language)[1]
            ar_input_content_ids = torch.tensor(
                [ar_input_content_ids], dtype=torch.long
            ).to(self.device)

            if style_ref_wav_path is not None:
                if style_ref_wav_text_language is None:
                    style_ref_wav_text_language = "zh"

                style_ref_input_ids = g2p_(
                    style_ref_wav_text, style_ref_wav_text_language
                )[1]
                style_ref_input_ids = torch.tensor(
                    [style_ref_input_ids], dtype=torch.long
                ).to(self.device)
                ar_input_content_ids = torch.cat(
                    [style_ref_input_ids, ar_input_content_ids], dim=1
                )

                if task == "recognition-synthesis":
                    # This is for the style code extration
                    src_speech16k = torch.cat(
                        [style_ref_speech16k, src_speech16k], dim=1
                    )
                    src_speech = np.concatenate([style_ref_speech, src_speech], axis=0)

        else:
            assert task == "conversion"

            src_speech16k = torch.cat([style_ref_speech16k, src_speech16k], dim=1)
            src_speech = np.concatenate([style_ref_speech, src_speech], axis=0)

            # [1, T]
            ar_input_content_ids = self.extract_coco_codec(
                "content",
                src_speech16k,
                src_speech,
            )

            if self.ar_cfg.model.train_both_conversion_and_synthesis:
                # [Important] When traing both VC and TTS, the VC's input_ids should be shifted, since Llama use a unified codebook
                # ar_input_ids += self.ar_cfg.model.tts_input_vocab_size
                raise NotImplementedError(
                    "train_both_conversion_and_synthesis is not implemented"
                )

        if use_style_tokens_as_ar_input:
            ar_input_style_ids = self.extract_coco_codec(
                "style",
                src_speech16k,
                src_speech,
                frame_len_ratio=target_src_duration_ratio,  # when target_src_duration_ratio < 1, the features will be downsampled
            )
        else:
            ar_input_style_ids = None

        if display_audio:
            print("AR input_content_ids:", ar_input_content_ids.shape)
            print(
                "AR input_style_ids:",
                "N/A" if ar_input_style_ids is None else ar_input_style_ids.shape,
            )

        if style_ref_wav_path is not None:
            prompt_output_ids = self.extract_coco_codec(
                "content_style", style_ref_speech16k, style_ref_speech
            )  # [1, T]
        else:
            prompt_output_ids = None

        if display_audio:
            if prompt_output_ids is not None:
                print("Prompt output_ids:", prompt_output_ids.shape)
            else:
                print("No prompt output_ids")

        # [1, T]
        predicted_coco_codecs = self.ar_model.generate(
            input_content_ids=ar_input_content_ids,
            input_style_ids=ar_input_style_ids,
            prompt_output_ids=prompt_output_ids,
        )

        ## Diffusion ##
        timbre_ref_codecs = self.extract_coco_codec(
            "content_style",
            timbre_ref_speech16k,
            timbre_ref_speech,
        )  # [1, T]
        diffusion_input_codecs = torch.cat(
            [timbre_ref_codecs, predicted_coco_codecs], dim=1
        )

        # Prepare the condition for diffusion
        diffusion_cond = self.fmt_model.cond_emb(diffusion_input_codecs)  # [1, T, D]
        if self.fmt_model.do_resampling:
            # Align to the frame rate of Mels
            diffusion_cond = self.fmt_model.resampling_layers(
                diffusion_cond.transpose(1, 2)
            ).transpose(1, 2)

        timbre_ref_mels = self.extract_mel_feature(timbre_ref_speech24k)  # [1, T, D]

        # [1, T, D]
        predict_mel_feat = self.fmt_model.reverse_diffusion(
            cond=diffusion_cond,
            prompt=timbre_ref_mels,
            n_timesteps=flow_matching_steps,
        )

        ## Vocoder and Display ##
        # [1, 1, T] -> [1, T]
        synthesized_audio = (
            self.vocoder_model(predict_mel_feat.transpose(1, 2)).detach().cpu()
        )[0]
        if display_audio:
            # [T]
            audio = synthesized_audio.numpy()[0]
            display_audio_in_notebook(audio, rate=24000)

        return synthesized_audio

    def inference_vocoder_resynthesis(self, wav_path, display_audio=False):
        speech, speech24k, speech16k = load_wav(wav_path, self.device)
        if display_audio:
            print("Ground Truth audio:")
            display_audio_in_notebook(speech, rate=24000)

        mel = self.extract_mel_feature(speech24k)  # [1, T, D]
        audio = self.vocoder_model(mel.transpose(1, 2)).detach().cpu()[0]
        if display_audio:
            print("Resynthesized audio:")
            display_audio_in_notebook(audio, rate=24000)
        return audio
