# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
import torch.nn.functional as F
import safetensors
import librosa

from models.tts.metis.audio_tokenizer import AudioTokenizer
from models.tts.maskgct.maskgct_utils import build_t2s_model, build_s2a_model, g2p_
from models.tts.metis.metis_model import MetisStage1
from peft import LoraModel, LoraConfig

from huggingface_hub import hf_hub_download, snapshot_download

import langid


def build_metis_stage1(cfg, device, ft_type=None):
    if ft_type == "l2s" or (
        hasattr(cfg, "use_zero_gate_adapter") and not cfg.use_zero_gate_adapter
    ):
        use_zero_gate_adapter = False
    else:
        use_zero_gate_adapter = True
    metis_stage1 = MetisStage1(
        cfg=cfg,
        ft_type=ft_type,
        ft_cond_dim=cfg.cond_dim,
        use_zero_gate_adapter=use_zero_gate_adapter,
    )
    if hasattr(cfg, "use_lora") and cfg.use_lora:
        lora_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.01,
        )
        metis_stage1 = LoraModel(metis_stage1, lora_config, adapter_name="default")
    metis_stage1.eval()
    metis_stage1.to(device)
    return metis_stage1


def build_metis_stage1_base(cfg, device, ft_type=None):
    metis_stage1 = MetisStage1(cfg=cfg, ft_type=ft_type, ft_cond_dim=cfg.cond_dim)
    metis_stage1.eval()
    metis_stage1.to(device)
    return metis_stage1


def merge_lora_weights(cfg, base_model, lora_weights):
    """Merge LoRA weights into the base model.

    Args:
        base_model: MetisStage1 model instance
        lora_weights: dict of LoRA weights (from safetensors or torch.load)

    Returns:
        MetisStage1: Model with merged weights
    """
    if isinstance(base_model, LoraModel):
        base_model = base_model.model  # Get the underlying model if it's wrapped

    # Create temporary LoRA model
    lora_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.01,
    )

    temp_model = LoraModel(base_model, lora_config, adapter_name="default")

    # Load LoRA weights
    temp_model.load_state_dict(lora_weights, strict=False)

    # Merge weights
    merged_model = temp_model.merge_and_unload()

    return merged_model


def merge_adapter_weights(base_model, adapter_weights, device):
    """Merge adapter weights into the base model.

    Args:
        base_model: MetisStage1 model instance
        adapter_weights: dict of adapter weights (adapter name: cond_emb)
    """

    for name, param in base_model.named_parameters():
        if "cond_adapter" in name:
            # print(name)
            param.data = adapter_weights["model." + name]
            # to device
            param.data = param.data.to(device)

    return base_model


def extract_lora_weights(model):
    """Extract LoRA weights from a LoraModel.

    Args:
        model: A LoraModel instance

    Returns:
        dict: LoRA weights state dict
    """
    if not isinstance(model, LoraModel):
        raise ValueError("Model must be a LoraModel instance")

    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:  # Only save LoRA parameters
            lora_state_dict[name] = param.data.clone()

    return lora_state_dict


def extract_adapter_weights(model):
    """Extract adapter weights from a model

    Args:
        model: A model instance

    Returns:
        dict: adapter weights state dict (adapter name: cond_emb)
    """

    adapter_state_dict = {}
    for name, param in model.named_parameters():
        if "cond_adapter" in name:
            adapter_state_dict[name] = param.data.clone()

    return adapter_state_dict


def bulid_visual_encoder(cfg, device):
    from models.tts.metis.vis_encoder import InferencePipeline

    visual_encoder = InferencePipeline(
        "video", cfg.vis_model_path, cfg.vis_model_conf, face_track=True, device=device
    )
    return visual_encoder


class Metis:

    def __init__(
        self,
        ckpt_path=None,
        base_ckpt_path=None,
        lora_ckpt_path=None,
        adapter_ckpt_path=None,
        cfg=None,
        device="cuda",
        model_type=None,  # support ["tts", "vc", "se", "tse", "l2s", "mix"]
    ):

        self.ckpt_path = ckpt_path
        self.cfg = cfg
        self.device = device
        self.model_type = model_type

        self.audio_tokenizer = AudioTokenizer(cfg, device)

        self.s2a_model_1layer, self.s2a_model_full = self._build_s2a_model()

        if ckpt_path is not None:
            self.metis_stage1 = build_metis_stage1(
                cfg.model.t2s_model, device, model_type
            )
            safetensors.torch.load_model(self.metis_stage1, ckpt_path)
            if model_type == "l2s":
                self.visual_encoder = bulid_visual_encoder(cfg.model.t2s_model, device)
        else:
            self.metis_stage1 = build_metis_stage1_base(
                cfg.model.t2s_model, device, ft_type=model_type
            )
            safetensors.torch.load_model(
                self.metis_stage1, base_ckpt_path, strict=False
            )
            print("load base model")

            # load adapter weights
            adapter_weights = safetensors.torch.load_file(adapter_ckpt_path)
            self.metis_stage1 = merge_adapter_weights(
                self.metis_stage1, adapter_weights, device
            )
            print("load adapter weights")

            # load lora weights
            lora_weights = safetensors.torch.load_file(lora_ckpt_path)
            self.metis_stage1 = merge_lora_weights(
                cfg.model.t2s_model, self.metis_stage1, lora_weights
            )
            print("load lora weights")

    @torch.no_grad()
    def __call__(
        self,
        text: str = None,
        prompt_speech_path: str = None,
        source_speech_path: str = None,  # used for se, tse, vc
        source_video_path: str = None,  # used for l2s
        prompt_text: str = None,
        prompt_language: str = None,
        target_language: str = None,
        target_len=None,  # in seconds
        n_timesteps: int = 25,
        cfg: float = 2.5,
        halton_scheduler: bool = False,
        model_type: str = "tts",
    ):
        if prompt_speech_path is not None:
            prompt_semantic_code, prompt_acoustic_code = self.preprocess_prompt_wav(
                prompt_speech_path
            )
        else:
            prompt_semantic_code, prompt_acoustic_code = None, None

        if model_type == "tts":
            combine_semantic_code, prompt_semantic_code = self.text2semantic(
                text,
                prompt_text,
                prompt_language,
                target_language,
                prompt_semantic_code,
                target_len,
                n_timesteps,
                cfg,
                halton_scheduler,
            )
        elif model_type == "se":
            source_speech_16k = librosa.load(source_speech_path, sr=16000)[0]
            combine_semantic_code = self.speech2semantic_wo_prompt(
                source_speech_16k, n_timesteps, cfg=cfg
            )
        elif model_type in ["vc", "tse"]:
            source_speech_16k = librosa.load(source_speech_path, sr=16000)[0]
            prompt_speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]
            combine_semantic_code = self.speech2semantic_w_prompt(
                source_speech_16k,
                prompt_speech_16k,
                prompt_semantic_code,
                n_timesteps,
                cfg=cfg,
            )
        elif model_type == "l2s":
            video_feature = self.visual_encoder.extract_features(source_video_path)
            combine_semantic_code = self.video2semantic(
                prompt_semantic_code, video_feature, n_timesteps, cfg
            )

        predict_acoustic_code = self.semantic2acoustic(
            combine_semantic_code, prompt_acoustic_code
        )

        predict_wav = self.audio_tokenizer.code2wav(predict_acoustic_code)

        return predict_wav

    def _build_s2a_model(self):

        # build s2a model
        s2a_model_1layer = build_s2a_model(
            self.cfg.model.s2a_model.s2a_1layer, self.device
        )
        s2a_model_full = build_s2a_model(self.cfg.model.s2a_model.s2a_full, self.device)

        # download s2a model
        s2a_1layer_dir = snapshot_download(
            repo_id="amphion/MaskGCT",
            repo_type="model",
            local_dir="./models/tts/metis/ckpt",
            allow_patterns=["s2a_model/s2a_model_1layer/model.safetensors"],
        )
        s2a_full_dir = snapshot_download(
            repo_id="amphion/MaskGCT",
            repo_type="model",
            local_dir="./models/tts/metis/ckpt",
            allow_patterns=["s2a_model/s2a_model_full/model.safetensors"],
        )
        s2a_1layer_ckpt = os.path.join(
            s2a_1layer_dir, "s2a_model/s2a_model_1layer/model.safetensors"
        )
        s2a_full_ckpt = os.path.join(
            s2a_full_dir, "s2a_model/s2a_model_full/model.safetensors"
        )

        # load s2a model
        safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
        safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

        s2a_model_1layer.eval()
        s2a_model_full.eval()

        return s2a_model_1layer, s2a_model_full

    @torch.no_grad()
    def preprocess_prompt_wav(self, prompt_speech_path: str):
        prompt_semantic_code, _, prompt_acoustic_code = self.audio_tokenizer(
            speech_path=prompt_speech_path
        )
        return prompt_semantic_code, prompt_acoustic_code

    @torch.no_grad()
    def semantic2acoustic(self, combine_semantic_code, acoustic_code):

        if acoustic_code is None:  # if no prompt
            acoustic_code = torch.zeros(1, 0, 12).to(self.device).long()

        semantic_code = combine_semantic_code
        cond = self.s2a_model_1layer.cond_emb(semantic_code)

        prompt = acoustic_code[:, :, :]
        predict_1layer = self.s2a_model_1layer.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=[40],
            cfg=0,
            rescale_cfg=0.75,
        )

        cond = self.s2a_model_full.cond_emb(semantic_code)

        prompt = acoustic_code[:, :, :]

        predict_full = self.s2a_model_full.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            temp=1.5,
            filter_thres=0.98,
            n_timesteps=[40, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            cfg=2.5,
            rescale_cfg=0.75,
            gt_code=predict_1layer,
        )

        return predict_full

    def tokenizer(
        self, text: str, prompt_text: str, prompt_language=None, target_language=None
    ):
        if prompt_language is None:
            prompt_language = langid.classify(prompt_text)[0]

        if target_language is None:
            target_language = langid.classify(text)[0]

        if prompt_language not in ["zh", "en", "ja", "fr", "ko", "de"]:
            prompt_language = "en"

        if target_language not in ["zh", "en", "ja", "fr", "ko", "de"]:
            target_language = "en"

        prompt_phone_id = g2p_(prompt_text, prompt_language)[1]
        target_phone_id = g2p_(text, target_language)[1]
        prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(
            self.device
        )
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(
            self.device
        )
        phone_id = torch.cat([prompt_phone_id, target_phone_id])
        return phone_id, prompt_phone_id, target_phone_id

    @torch.no_grad()
    def speech2semantic_wo_prompt(self, speech: np.ndarray, steps=10, cfg=0):
        semantic_feat = self.audio_tokenizer.wav2semantic_feat(speech)
        predict_semantic_code = self.metis_stage1.reverse_diffusion(
            torch.zeros(1, 0).to(self.device).long(),
            semantic_feat.shape[1],
            None,
            n_timesteps=steps,
            cfg=cfg,
            rescale_cfg=0.75,
            finetune_cond=semantic_feat,
        )
        return predict_semantic_code

    @torch.no_grad()
    def speech2semantic_w_prompt(
        self,
        speech: np.ndarray,
        prompt_speech: np.ndarray,
        prompt_semantic_code: torch.Tensor,
        steps=10,
        cfg=0,
    ):
        semantic_feat = self.audio_tokenizer.wav2semantic_feat(
            np.concatenate([prompt_speech, speech])
        )

        target_len = semantic_feat.shape[1] - prompt_semantic_code.shape[1]
        predict_semantic_code = self.metis_stage1.reverse_diffusion(
            prompt_semantic_code,
            target_len,
            None,
            n_timesteps=steps,
            cfg=cfg,
            rescale_cfg=0.75,
            finetune_cond=semantic_feat,
        )

        combine_semantic_code = torch.cat(
            [prompt_semantic_code, predict_semantic_code], dim=-1
        )

        return combine_semantic_code

    @torch.no_grad()
    def text2semantic(
        self,
        text,
        prompt_text,
        prompt_language,
        target_language,
        prompt_semantic_code,
        target_len=None,
        n_timesteps=25,
        cfg=2.5,
        halton_scheduler=False,
    ):

        phone_id, prompt_phone_id, target_phone_id = self.tokenizer(
            text, prompt_text, prompt_language, target_language
        )

        if target_len is None:
            target_len = (
                prompt_semantic_code.shape[1]
                * len(text.encode("utf-8"))
                // len(prompt_text.encode("utf-8"))
            )
        else:
            target_len = int(target_len * 50)  # 50 tokens per second

        # TODO: halton scheduler
        # if halton_scheduler:
        #     preschedule_mask_indices = discrete_halton_sequence(2, target_len)
        # else:
        #     preschedule_mask_indices = None

        predict_semantic = self.metis_stage1.reverse_diffusion(
            prompt_semantic_code[:, :],
            target_len,
            phone_id.unsqueeze(0),
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=0.75,
            # preschedule_mask_indices=preschedule_mask_indices,
        )

        combine_semantic_code = torch.cat(
            [prompt_semantic_code[:, :], predict_semantic], dim=-1
        )

        return combine_semantic_code, prompt_semantic_code

    @torch.no_grad()
    def video2semantic(
        self, prompt_semantic_code, video_feature, n_timesteps=25, cfg=0
    ):

        video_feature = video_feature.unsqueeze(0).transpose(1, 2)
        video_feature = F.interpolate(
            video_feature, scale_factor=2, mode="linear", align_corners=False
        )
        video_feature = video_feature.transpose(1, 2).squeeze(0)

        target_len = video_feature.shape[0]
        prompt_len = prompt_semantic_code.shape[-1]

        if prompt_len > 0:
            zeros_tensor = torch.zeros((prompt_len, video_feature.shape[-1])).to(
                self.device
            )
            video_feature = torch.cat((zeros_tensor, video_feature), dim=0)
        video_feature = video_feature.unsqueeze(0)

        predict_semantic_code = self.metis_stage1.reverse_diffusion(
            prompt_semantic_code,
            target_len,
            None,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=0.75,
            finetune_cond=video_feature,
        )

        combine_semantic_code = torch.cat(
            [prompt_semantic_code, predict_semantic_code], dim=-1
        )

        return combine_semantic_code
