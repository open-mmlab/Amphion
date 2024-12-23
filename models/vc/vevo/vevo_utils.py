# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import librosa
import torch
import torchaudio
import accelerate
import safetensors
import numpy as np
import yaml
from IPython.display import display, Audio

from models.vc.flow_matching_transformer.fmt_model import FlowMatchingTransformer
from models.vc.autoregressive_transformer.ar_model import AutoregressiveTransformer
from models.codec.kmeans.repcodec_model import RepCodec
from models.codec.vevo.vevo_repcodec import VevoRepCodec
from models.codec.melvqgan.melspec import MelSpectrogram
from models.codec.amphion_codec.vocos import Vocos

from utils.util import load_config


def g2p_(text, language):
    from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p

    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return g2p(text, sentence=None, language=language)


def transcribe_audio(audio_path, model=None):
    if model is None:
        import whisper

        model = whisper.load_model("medium")

    result = model.transcribe(audio_path)
    return result["text"]


# Semantic Features Extractor
def build_hubert_model(device):
    bundle = torchaudio.pipelines.HUBERT_LARGE
    hubert = bundle.get_model()
    hubert.eval()
    hubert.to(device)
    return hubert


# VQ-VAE Tokenizer
def build_vqvae_model(repcodec_cfg, device):
    vqvae = RepCodec(cfg=repcodec_cfg)
    vqvae.eval()
    vqvae.to(device)
    return vqvae


# Vevo VQ-VAE Tokenizer (pkl checkpoint)
def load_vevo_vqvae_checkpoint(repcodec_cfg, device):
    with open(repcodec_cfg.config_path) as fp:
        conf = yaml.load(fp, Loader=yaml.FullLoader)
    vqvae = VevoRepCodec(**conf)
    vqvae.quantizer.initial()
    vqvae.eval()

    pretrained_path = repcodec_cfg.pretrained_path
    if ".pkl" in pretrained_path:
        # Vevo paper
        vqvae.load_state_dict(
            torch.load(pretrained_path, map_location="cpu")["model"]["repcodec"]
        )
    elif ".safetensors" in pretrained_path:
        # Re-trained vevovq
        safetensors.torch.load_model(vqvae, pretrained_path)

    vqvae.to(device)
    return vqvae


# Autoregressive Transformer
def build_ar_model(cfg, device):
    model = AutoregressiveTransformer(cfg=cfg.model.autoregressive_transformer)
    model.eval()
    model.to(device)
    return model


# Flow Matching Transformer
def build_fmt_model(cfg, device):
    model = FlowMatchingTransformer(cfg=cfg.model.flow_matching_transformer)
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


class VevoInferencePipeline:
    def __init__(
        self,
        content_tokenizer_ckpt_path=None,
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

        if ar_cfg_path is not None and ar_ckpt_path is not None:
            self.ar_cfg = load_config(ar_cfg_path)
            self.ar_model = load_checkpoint(
                build_ar_model, self.ar_cfg, ar_ckpt_path, device
            )
            print(f"#Params of AR model: {count_parameters(self.ar_model)}")
        else:
            self.ar_cfg = None
            self.ar_model = None

        self.fmt_cfg = load_config(fmt_cfg_path)
        self.fmt_model = load_checkpoint(
            build_fmt_model, self.fmt_cfg, fmt_ckpt_path, device
        )
        print(f"#Params of Flow Matching model: {count_parameters(self.fmt_model)}")

        self.vocoder_cfg = load_config(vocoder_cfg_path)
        self.mel_model = build_mel_model(self.vocoder_cfg, device)
        self.vocoder_model = load_checkpoint(
            build_vocoder_model, self.vocoder_cfg, vocoder_ckpt_path, device
        )
        print(f"#Params of Vocoder model: {count_parameters(self.vocoder_model)}")

        self.content_tokenizer_ckpt_path = content_tokenizer_ckpt_path
        self.content_style_tokenizer_ckpt_path = content_style_tokenizer_ckpt_path
        self.init_vqvae_tokenizer()

    def init_vqvae_tokenizer(self):
        ## HuBERT features extraction ##
        self.hubert_model = build_hubert_model(self.device)
        stat = np.load(self.fmt_cfg.model.representation_stat_mean_var_path)
        self.hubert_feat_norm_mean = torch.tensor(stat["mean"])
        self.hubert_feat_norm_std = torch.tensor(stat["std"])

        ## Content Tokenizer ##
        if self.ar_model is not None and "input_repcodec" in self.ar_cfg.model:
            assert self.ar_cfg.model.vc_input_token_type == "hubert_vevo_codec"

            ckpt_path = getattr(
                self.ar_cfg.model.input_repcodec,
                "pretrained_path",
                self.content_tokenizer_ckpt_path,
            )
            self.ar_cfg.model.input_repcodec.pretrained_path = ckpt_path
            self.content_tokenizer = load_vevo_vqvae_checkpoint(
                self.ar_cfg.model.input_repcodec,
                self.device,
            )

            print(
                "#Params of Content Tokenizer: {}".format(
                    count_parameters(self.content_tokenizer)
                )
            )

        ## Content-Style Tokenizer ##
        ckpt_path = getattr(
            self.fmt_cfg.model.repcodec,
            "pretrained_path",
            self.content_style_tokenizer_ckpt_path,
        )
        self.content_style_tokenizer = load_checkpoint(
            build_vqvae_model,
            self.fmt_cfg.model.repcodec,
            ckpt_path,
            self.device,
        )
        print(
            "#Params of Content-Style Tokenizer: {}".format(
                count_parameters(self.content_style_tokenizer)
            )
        )

    @torch.no_grad()
    def extract_mel_feature(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.vocoder_cfg.preprocess.mel_mean) / math.sqrt(
            self.vocoder_cfg.preprocess.mel_var
        )
        return mel_feature

    @torch.no_grad()
    def extract_prompt_mel_feature(self, speech):
        """
        This is for the global encoder of AR model
        """
        if not hasattr(self, "prompt_mel_model"):
            self.prompt_mel_model = build_mel_model(self.ar_cfg, self.device)

        mel_feature = self.prompt_mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.ar_cfg.preprocess.mel_mean) / math.sqrt(
            self.ar_cfg.preprocess.mel_var
        )
        return mel_feature

    @torch.no_grad()
    def extract_hubert_feature(self, wavs, wav_lens=None, output_layer=18):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            feats: [B, T, D]
            feat_lengths: [B]
        """
        if wav_lens is None:
            wav_lens = torch.tensor([wavs.shape[1]] * wavs.shape[0]).to(wavs).int()

        feats, feat_lengths = self.hubert_model.extract_features(
            wavs, lengths=wav_lens, num_layers=output_layer
        )
        feats = feats[-1]
        return feats, feat_lengths

    def duration_reduction_func(self, token_seq, n_gram=1):
        """
        Args:
            token_seq: (T,)
        Returns:
            reduced_token_seq: (T')
            reduced_token_seq_len: T'
        """
        n_gram_seq = token_seq.unfold(0, n_gram, 1)
        mask = torch.all(n_gram_seq[1:] != n_gram_seq[:-1], dim=1)
        reduced_token_seq = torch.cat(
            (n_gram_seq[0, :n_gram], n_gram_seq[1:, -1][mask])
        )
        return reduced_token_seq, len(reduced_token_seq)

    @torch.no_grad()
    def extract_hubert_codec(
        self,
        vqvae_model,
        wavs,
        wav_lens=None,
        output_layer=18,
        token_type="hubert_codec",
        duration_reduction=False,
        duration_reduction_n_gram=1,
    ):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            codecs: [B, T]
            codec_masks: [B, T]
        """
        # Extract features and normalize
        feats, feat_lengths = self.extract_hubert_feature(wavs, wav_lens, output_layer)

        if token_type == "hubert_codec":
            feats = (
                feats - self.hubert_feat_norm_mean.to(feats)
            ) / self.hubert_feat_norm_std.to(feats)
            codecs, _ = vqvae_model.quantize(feats)  # (B, T)
        elif token_type == "hubert_vevo_codec":
            x = vqvae_model.encoder(feats.transpose(1, 2))
            z = vqvae_model.projector(x)
            _, idx = vqvae_model.quantizer.codebook.forward_index(z.transpose(2, 1))
            codecs = idx[0]  # (B, T)
        else:
            raise ValueError("Invalid token_type")

        if not duration_reduction:
            T = codecs.shape[1]
            arange_tensor = torch.arange(T).expand(codecs.shape[0], T).to(codecs)
            codec_masks = (
                arange_tensor < feat_lengths.unsqueeze(-1)
            ).int()  # 1 means valid
            return codecs, codec_masks

        else:
            reduced_codecs = []
            reduced_masks = []

            for i, token_seq_len in enumerate(feat_lengths):
                token_seq = codecs[i, :token_seq_len]
                reduced_token_seq, reduced_token_seq_len = self.duration_reduction_func(
                    token_seq, n_gram=duration_reduction_n_gram
                )

                reduced_codecs.append(reduced_token_seq)
                reduced_masks.append(
                    torch.ones(reduced_token_seq_len, dtype=torch.int).to(codecs)
                )

            reduced_codecs = torch.nn.utils.rnn.pad_sequence(
                reduced_codecs, batch_first=True, padding_value=0
            )
            reduced_masks = torch.nn.utils.rnn.pad_sequence(
                reduced_masks, batch_first=True, padding_value=0
            )
            return reduced_codecs, reduced_masks

    def random_mask_codec(self, codecs, codec_masks, ratio, mask_value):
        """
        Args:
            codecs: [B, T]
            codec_masks: [B, T], 0 means not to mask
            ratio: float
            mask_value: int
        Returns:
            masked_codecs: [B, T]
        """
        rand_mask = (torch.rand_like(codecs.float(), device=codecs.device) < ratio) & (
            codec_masks == 1
        )
        masked_codecs = codecs.masked_fill(rand_mask, mask_value)
        return masked_codecs

    def inference_ar_and_fm(
        self,
        src_wav_path,
        src_text,
        style_ref_wav_path,
        timbre_ref_wav_path,
        style_ref_wav_text=None,
        src_text_language=None,
        style_ref_wav_text_language=None,
        vc_input_mask_ratio=-1,
        use_global_guided_inference=False,
        flow_matching_steps=32,
        display_audio=False,
    ):
        assert self.ar_model is not None

        if src_wav_path is None:
            # TTS
            task = "tts"
            assert src_text is not None

            if src_text_language is None:
                src_text_language = "zh"
            if style_ref_wav_text_language is None:
                style_ref_wav_text_language = "zh"

            if display_audio:
                print("-" * 20)
                print("Source Text: [{}]".format(src_text))

        else:
            # VC
            task = "vc"
            assert src_text is None
            src_speech, src_speech24k, src_speech16k = load_wav(
                src_wav_path, self.device
            )

            if display_audio:
                print("-" * 20)
                print("Source audio:")
                display_audio_in_notebook(src_speech, rate=24000)

        style_ref_speech, style_ref_speech24k, style_ref_speech16k = load_wav(
            style_ref_wav_path, self.device
        )
        timbre_ref_speech, timbre_ref_speech24k, timbre_ref_speech16k = load_wav(
            timbre_ref_wav_path, self.device
        )

        if display_audio:
            if style_ref_wav_path == timbre_ref_wav_path:
                print("Both Style and Timbre Reference audio:")
                display_audio_in_notebook(style_ref_speech, rate=24000)
            else:
                print("Style Reference audio:")
                display_audio_in_notebook(style_ref_speech, rate=24000)
                print("Timbre Reference audio:")
                display_audio_in_notebook(timbre_ref_speech, rate=24000)
                print("-" * 20)

        ## AR ##
        if task == "tts":
            ar_input_ids = g2p_(src_text, src_text_language)[1]
            ar_input_ids = torch.tensor([ar_input_ids], dtype=torch.long).to(
                self.device
            )

            if display_audio:
                print("Src text input_ids:", ar_input_ids.shape)

            if not use_global_guided_inference:
                assert style_ref_wav_text is not None
                style_ref_input_ids = g2p_(
                    style_ref_wav_text, style_ref_wav_text_language
                )[1]
                style_ref_input_ids = torch.tensor(
                    [style_ref_input_ids], dtype=torch.long
                ).to(self.device)
                ar_input_ids = torch.cat([style_ref_input_ids, ar_input_ids], dim=1)

                if display_audio:
                    print("AR input_ids:", ar_input_ids.shape)

        elif task == "vc":
            if not use_global_guided_inference:
                src_speech16k = torch.cat([style_ref_speech16k, src_speech16k], dim=1)

            # [1, T]
            ar_input_ids, _ = self.extract_hubert_codec(
                self.content_tokenizer,
                src_speech16k,
                token_type=self.ar_cfg.model.vc_input_token_type,
                duration_reduction=True,
                duration_reduction_n_gram=getattr(
                    self.ar_cfg.model, "vc_input_reduced_n_gram", 1
                ),
            )

            if vc_input_mask_ratio > 0:
                ar_input_masks = torch.ones_like(
                    ar_input_ids, dtype=torch.int, device=self.device
                )
                if not use_global_guided_inference:
                    total_len = ar_input_ids.shape[1]
                    style_ref_ratio = (
                        style_ref_speech16k.shape[1] / src_speech16k.shape[1]
                    )
                    ar_input_masks[:, : int(total_len * style_ref_ratio)] = 0

                ar_input_ids = self.random_mask_codec(
                    codecs=ar_input_ids,
                    codec_masks=ar_input_masks,
                    ratio=vc_input_mask_ratio,
                    mask_value=self.ar_cfg.model.vc_input_vocab_size,
                )

            if self.ar_cfg.model.train_both_vc_and_tts:
                # [Important] When traing both VC and TTS, the VC's input_ids should be shifted, since Llama use a unified codebook
                ar_input_ids += self.ar_cfg.model.tts_input_vocab_size

            if display_audio:
                print("AR input_ids:", ar_input_ids.shape)

        if use_global_guided_inference:
            prompt_output_ids = None
        else:
            prompt_output_ids, _ = self.extract_hubert_codec(
                self.content_style_tokenizer,
                style_ref_speech16k,
                duration_reduction=False,
            )
            if display_audio:
                print("Prompt output_ids:", prompt_output_ids.shape)

        # [1, T]
        predicted_hubert_codecs = self.ar_model.generate(
            input_ids=ar_input_ids,
            prompt_mels=self.extract_prompt_mel_feature(style_ref_speech16k),
            prompt_output_ids=prompt_output_ids,
        )

        ## Diffusion ##
        timbre_ref_hubert_codecs, _ = self.extract_hubert_codec(
            self.content_style_tokenizer, timbre_ref_speech16k, duration_reduction=False
        )
        diffusion_input_codecs = torch.cat(
            [timbre_ref_hubert_codecs, predicted_hubert_codecs], dim=1
        )

        # [1, T, D]
        predict_mel_feat = self.fmt_model.reverse_diffusion(
            cond=self.fmt_model.cond_emb(diffusion_input_codecs),
            prompt=self.extract_mel_feature(timbre_ref_speech24k),
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

    def inference_fm(
        self,
        src_wav_path,
        timbre_ref_wav_path,
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
                print("Audio:")
                display_audio_in_notebook(src_wav_path, rate=24000)
            else:
                print("Source audio:")
                display_audio_in_notebook(src_speech, rate=24000)
                print("Timbre Reference audio:")
                display_audio_in_notebook(timbre_ref_speech, rate=24000)
                print("-" * 20)

        ## Diffusion ##
        src_hubert_codecs, _ = self.extract_hubert_codec(
            self.content_style_tokenizer, src_speech16k, duration_reduction=False
        )
        timbre_ref_hubert_codecs, _ = self.extract_hubert_codec(
            self.content_style_tokenizer, timbre_ref_speech16k, duration_reduction=False
        )
        diffusion_input_codecs = torch.cat(
            [timbre_ref_hubert_codecs, src_hubert_codecs], dim=1
        )

        # [1, T, D]
        predict_mel_feat = self.fmt_model.reverse_diffusion(
            cond=self.fmt_model.cond_emb(diffusion_input_codecs),
            prompt=self.extract_mel_feature(timbre_ref_speech24k),
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
