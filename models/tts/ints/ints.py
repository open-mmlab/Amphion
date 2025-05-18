# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from huggingface_hub import snapshot_download, hf_hub_download
import torch
import torch.nn.functional as F
import librosa
import math
import accelerate
import safetensors
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.tts.ints.chat_template import (
    gen_chat_prompt_for_tts,
)
from models.tts.ints.tokenizer import IntsAudioTokenizer
from models.tts.voicebox.voicebox_model import VoiceBox
from models.codec.melvqgan.melspec import MelSpectrogram
from models.codec.amphion_codec.vocos import Vocos

try:
    from vllm import LLM, SamplingParams
except:
    print("vllm is not installed, please install it first if you want to use vllm.")
    pass


def build_vocoder_model(cfg, device):
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    vocoder_model = Vocos(cfg=cfg.model.vocos)
    vocoder_model.eval()
    vocoder_model.to(device)
    accelerate.load_checkpoint_and_dispatch(vocoder_model, vocoder_ckpt_path)
    return vocoder_model


def build_voicebox(cfg, device):
    voicebox_ckpt_path = hf_hub_download("amphion/Ints", "voicebox/model.safetensors")
    print(f"voicebox_ckpt_path: {voicebox_ckpt_path}")

    soundstorm_model = VoiceBox(cfg=cfg.model.voicebox)
    soundstorm_model.eval()
    soundstorm_model.to(device)
    safetensors.torch.load_model(soundstorm_model, voicebox_ckpt_path)
    return soundstorm_model


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


class Ints:
    def __init__(
        self,
        llm_path,
        cfg,
        device="cuda",
        w2v_bert_path="facebook/w2v-bert-2.0",
        build_llm=True,
        use_vllm=False,
        use_flash_attn=False,
        gpu_memory_utilization=0.4,
    ):
        self.cfg = cfg
        self.device = torch.device(device)

        # llm
        self.llm_path = llm_path

        # use vllm or not
        self.use_vllm = use_vllm

        print(f"use_vllm: {self.use_vllm}")
        self.llm = None
        if build_llm:
            self.build_llm_model(
                llm_path, device, use_vllm, use_flash_attn, gpu_memory_utilization
            )

        # text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path)

        # audio tokenizer
        self.audio_tokenizer = IntsAudioTokenizer(cfg, w2v_bert_path, device)

        # vocoder
        self.vocoder = build_vocoder_model(cfg, device)

        # voicebox
        self.voicebox = build_voicebox(cfg, device)

        # mel model
        self.mel_model = build_mel_model(cfg, device)

    def __call__(
        self,
        target_text: str,
        prompt_speech_path: str,
        prompt_text: str,
        top_k: int = 20,
        top_p: float = 0.98,
        temp: float = 1.0,
    ):
        prompt_semantic_code, prompt_speech = self.preprocess_prompt_wav(
            prompt_speech_path
        )

        debug_dict = {
            "prompt_semantic_code_length": prompt_semantic_code.size(1),
            "prompt_text": prompt_text,
            "target_text": target_text,
            "use_vllm": self.use_vllm,
        }

        if not self.use_vllm:

            input_ids = self.tokenize(
                prompt_text,
                target_text,
                prompt_semantic_code,
            )

            debug_dict["input_ids_length"] = input_ids.size(1)

            generate_ids = self.llm.generate(
                input_ids=input_ids,
                min_new_tokens=12,
                max_new_tokens=400,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temp,
            )

            debug_dict["generate_ids_length"] = generate_ids.size(1)

            (
                prompt_semantic_code,
                generate_semantic_code,
                combine_semantic_code,
            ) = self.postprocess(prompt_semantic_code, generate_ids, input_ids.size(1))

            debug_dict["combine_semantic_code_length"] = combine_semantic_code.size(1)

        else:
            prompts = [
                gen_chat_prompt_for_tts(prompt_text + target_text)
                + self.tensor_to_audio_string(prompt_semantic_code)
            ]

            debug_dict["prompts"] = prompts

            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                skip_special_tokens=False,
            )

            debug_dict["sampling_params"] = {
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_k": sampling_params.top_k,
                "top_p": sampling_params.top_p,
                "skip_special_tokens": sampling_params.skip_special_tokens,
            }

            outputs = self.llm.generate(prompts, sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text

            debug_dict["generated_text"] = generated_text

            generate_semantic_code = self.extract_audio_ids(generated_text)
            generate_semantic_code = (
                torch.tensor(generate_semantic_code).to(self.device).unsqueeze(0)
            )

            combine_semantic_code = torch.cat(
                [prompt_semantic_code, generate_semantic_code], dim=1
            )

        predict_mel = self.code2mel(combine_semantic_code, prompt_speech)
        audio = self.mel2audio(predict_mel)

        return audio, debug_dict

    def build_llm_model(
        self, llm_path, device, use_vllm, use_flash_attn, gpu_memory_utilization=0.4
    ):
        if use_vllm:
            self.llm = LLM(
                model=llm_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=768,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=(
                    "flash_attention_2" if use_flash_attn else "eager"
                ),
            )

    def decode_audio_ids(self, audio_ids: list):
        audio_ids = self.extract_audio_ids(audio_ids)
        audio_ids = torch.tensor(audio_ids).to(self.device).unsqueeze(0)
        predict_mel = self.code2mel(audio_ids, None)
        audio = self.mel2audio(predict_mel)
        return audio

    def preprocess_prompt_wav(self, prompt_speech_path: str):
        speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]
        speech = librosa.load(prompt_speech_path, sr=24000)[0]
        semantic_code = self.audio_tokenizer(speech_16k)
        return semantic_code, speech

    def tokenize(
        self, prompt_text: str, target_text: str, prompt_semantic_code: torch.Tensor
    ):
        text = gen_chat_prompt_for_tts(prompt_text + target_text)
        text_token_ids = self.tokenizer(text, return_tensors="pt").to(self.device)

        # shift speech ids
        prompt_semantic_code = prompt_semantic_code + 32066
        # add bos audio token
        prompt_semantic_code = torch.cat(
            [torch.tensor([[32064]], device=self.device), prompt_semantic_code], dim=1
        )

        input_ids = torch.cat([text_token_ids.input_ids, prompt_semantic_code], dim=1)

        return input_ids

    def postprocess(
        self,
        prompt_semantic_code: torch.Tensor,
        generate_ids: torch.Tensor,
        input_ids_size: int,
    ):
        generate_semantic_code = generate_ids[:, input_ids_size:]
        generate_semantic_code = generate_semantic_code[:, :-2]
        generate_semantic_code = generate_semantic_code - 32066
        combine_semantic_code = torch.cat(
            [prompt_semantic_code, generate_semantic_code], dim=1
        )

        # assert combine_semantic_code in [0, 16384), else random replace the token not in [0, 16384)
        for i in range(combine_semantic_code.shape[0]):
            for j in range(combine_semantic_code.shape[1]):
                if (
                    combine_semantic_code[i, j] < 0
                    or combine_semantic_code[i, j] >= 16384
                ):
                    combine_semantic_code[i, j] = torch.randint(0, 16384, (1,))

        return prompt_semantic_code, generate_semantic_code, combine_semantic_code

    def code2mel(self, combine_semantic_code: torch.Tensor, prompt_speech):
        cond_feature = self.voicebox.cond_emb(combine_semantic_code)
        cond_feature = F.interpolate(
            cond_feature.transpose(1, 2),
            scale_factor=self.voicebox.cond_scale_factor,
        ).transpose(1, 2)

        if prompt_speech is not None:
            prompt_mel_feat = self.extract_mel_feature(
                torch.tensor(prompt_speech).to(self.device).unsqueeze(0),
            )
        else:
            prompt_mel_feat = None

        predict_mel = self.voicebox.reverse_diffusion(
            cond_feature,
            prompt_mel_feat,
            n_timesteps=32,
            cfg=2.0,
            rescale_cfg=0.75,
        )

        return predict_mel

    def mel2audio(self, mel_feature: torch.Tensor):
        audio = self.vocoder(mel_feature.transpose(1, 2)).detach().cpu().numpy()[0][0]
        return audio

    @torch.no_grad()
    def extract_mel_feature(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.cfg.preprocess.mel_mean) / math.sqrt(
            self.cfg.preprocess.mel_var
        )
        return mel_feature

    def tensor_to_audio_string(self, tensor):
        """
        Convert a tensor of shape [[11570, 13970, 2667, ...]] to a string format
        "<|audio_11570|><|audio_13970|><|audio_2667|>..."

        Args:
            tensor: A 2D tensor or list containing integers

        Returns:
            Formatted string
        """
        # Ensure we're working with a 1D array (take the first row)
        if isinstance(tensor, list) and isinstance(tensor[0], list):
            values = tensor[0]
        else:
            values = tensor[0].tolist() if hasattr(tensor, "tolist") else tensor[0]

        result = ""
        result += "<|start_of_audio|>"
        for value in values:
            result += f"<|audio_{value}|>"

        return result

    def extract_audio_ids(self, text):
        """
        Extract all audio IDs from a string containing <|audio_X|> markers

        Args:
            text (str): A string containing audio markers

        Returns:
            list: A list of all audio IDs
        """
        import re

        # Use regex to match all <|audio_X|> format markers
        pattern = r"<\|audio_(\d+)\|>"

        # Find all matches and extract the numeric part
        audio_ids = re.findall(pattern, text)

        # Convert string IDs to integers
        audio_ids = [int(id) for id in audio_ids]

        return audio_ids
