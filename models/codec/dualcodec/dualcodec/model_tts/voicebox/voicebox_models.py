# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
from .llama_nar import DiffLlama
import torch.nn.functional as F


def module_size(module: nn.Module, trainable_only: bool = False) -> int:
    """
    Calculate the total number of parameters in a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module.
        trainable_only (bool): If True, only count trainable parameters.

    Returns:
        int: Total number of parameters in the module.
    """
    return sum(
        p.numel() for p in module.parameters() if not trainable_only or p.requires_grad
    )


class VoiceBox(nn.Module):
    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        cfg_scale=0.2,
        use_cond_code=True,
        cond_codebook_size=1024,
        cond_dim=1024,
        cond_scale_factor=1,
        sigma=1e-5,
        time_scheduler="linear",
        cfg=None,
    ):
        super().__init__()

        mel_dim = (
            cfg.mel_dim if cfg is not None and hasattr(cfg, "mel_dim") else mel_dim
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        num_layers = (
            cfg.num_layers
            if cfg is not None and hasattr(cfg, "num_layers")
            else num_layers
        )
        num_heads = (
            cfg.num_heads
            if cfg is not None and hasattr(cfg, "num_heads")
            else num_heads
        )
        cfg_scale = (
            cfg.cfg_scale
            if cfg is not None and hasattr(cfg, "cfg_scale")
            else cfg_scale
        )
        use_cond_code = (
            cfg.use_cond_code
            if cfg is not None and hasattr(cfg, "use_cond_code")
            else use_cond_code
        )
        cond_codebook_size = (
            cfg.cond_codebook_size
            if cfg is not None and hasattr(cfg, "cond_codebook_size")
            else cond_codebook_size
        )
        cond_dim = (
            cfg.cond_dim if cfg is not None and hasattr(cfg, "cond_dim") else cond_dim
        )
        time_scheduler = (
            cfg.time_scheduler
            if cfg is not None and hasattr(cfg, "time_scheduler")
            else time_scheduler
        )
        sigma = cfg.sigma if cfg is not None and hasattr(cfg, "sigma") else sigma
        cond_scale_factor = (
            cfg.cond_scale_factor
            if cfg is not None and hasattr(cfg, "cond_scale_factor")
            else cond_scale_factor
        )

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.use_cond_code = use_cond_code
        self.cond_codebook_size = cond_codebook_size
        self.cond_dim = cond_dim
        self.time_scheduler = time_scheduler
        self.sigma = sigma
        self.cond_scale_factor = cond_scale_factor

        if self.use_cond_code:
            self.cond_emb = nn.Embedding(cond_codebook_size, self.hidden_size)
        else:
            self.cond_emb = nn.Linear(self.cond_dim, self.hidden_size)

        self.reset_parameters()

        self.diff_estimator = DiffLlama(
            mel_dim=mel_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.sigma = sigma

    @torch.no_grad()
    def forward_diffusion(self, x, t):
        """
        x: (B, T, mel_dim)
        t: (B,)
        """

        new_t = t
        t = t.unsqueeze(-1).unsqueeze(-1)
        z = torch.randn(
            x.shape, dtype=x.dtype, device=x.device, requires_grad=False
        )  # (B, T, mel_dim)

        cfg_scale = self.cfg_scale

        # get prompt len
        if torch.rand(1) > cfg_scale:
            prompt_len = torch.randint(
                min(x.shape[1] // 4, 5), int(x.shape[1] * 0.4), (x.shape[0],)
            ).to(
                x.device
            )  # (B,)
        else:
            prompt_len = torch.zeros(x.shape[0]).to(x)  # (B,)

        # get is prompt
        is_prompt = torch.zeros_like(x[:, :, 0])  # (B, T)
        col_indices = (
            torch.arange(is_prompt.shape[1])
            .repeat(is_prompt.shape[0], 1)
            .to(prompt_len)
        )  # (B, T)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # (B, T) 1 if prompt

        mask = torch.ones_like(x[:, :, 0])  # mask if 1, not mask if 0
        mask[is_prompt.bool()] = 0
        mask = mask[:, :, None]

        # flow matching: xt = (1 - (1 - sigma) * t) * x0 + t * x; where x0 ~ N(0, 1), x is a sample
        # flow gt: x - (1 - sigma) * x0 = x - (1 - sigma) * noise
        xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)

        return xt, z, new_t, prompt_len, mask

    def loss_t(
        self,
        x,
        x_mask,
        t,
        cond=None,
    ):
        xt, z, new_t, prompt_len, mask = self.forward_diffusion(x, t)

        noise = z

        # drop all condition for cfg, so if prompt_len is 0, we also drop cond
        if cond is not None:
            cond = cond * torch.where(
                prompt_len > 0,
                torch.ones_like(prompt_len),
                torch.zeros_like(prompt_len),
            ).to(cond.device).unsqueeze(-1).unsqueeze(-1)

        flow_pred = self.diff_estimator(xt, new_t, cond, x_mask)  # (B, T, mel_dim)

        # final mask used for loss calculation
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        return noise, x, flow_pred, final_mask, prompt_len

    def compute_loss(self, x, x_mask, cond=None):
        # x0: (B, T, num_quantizer)
        # x_mask: (B, T) mask is 0 for padding
        t = torch.rand(x.shape[0], device=x.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        # from CosyVoice: considering the generation process at the beginning is harder than follows, we involve a cosine scheduler for the timestep t
        if self.time_scheduler == "cos":
            t = 1 - torch.cos(t * math.pi * 0.5)
        else:
            pass
        return self.loss_t(x, x_mask, t, cond)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    @torch.no_grad()
    def reverse_diffusion(
        self,
        cond,
        prompt,
        x_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        h = 1.0 / n_timesteps

        if prompt is None:
            prompt = torch.zeros(cond.shape[0], 0, self.mel_dim).to(cond.device)

        prompt_len = prompt.shape[1]
        target_len = cond.shape[1] - prompt_len

        if x_mask == None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)  # (B, T)
        if prompt_mask == None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(
                cond.device
            )  # (B, prompt_len)
        xt_mask = torch.cat([prompt_mask, x_mask], dim=1)
        z = torch.randn(
            (cond.shape[0], target_len, self.mel_dim),
            dtype=cond.dtype,
            device=cond.device,
            requires_grad=False,
        )
        xt = z
        # t from 0 to 1: x0 = z ~ N(0, 1)
        for i in range(n_timesteps):
            xt_input = torch.cat([prompt, xt], dim=1)  # [b t c]
            t = (0 + (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            flow_pred = self.diff_estimator(xt_input, t, cond, xt_mask)
            flow_pred = flow_pred[:, prompt_len:, :]
            # cfg

            if cfg > 0:
                uncond_flow_pred = self.diff_estimator(
                    xt, t, torch.zeros_like(cond)[:, : xt.shape[1], :], x_mask
                )
                pos_flow_pred_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescale_flow_pred = (
                    flow_pred_cfg * pos_flow_pred_std / flow_pred_cfg.std()
                )
                flow_pred = (
                    rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg
                )

            dxt = flow_pred * h
            xt = xt + dxt

        return xt

    def forward(self, x, x_mask, cond_code=None, cond_feature=None):
        if cond_code != None and self.use_cond_code:
            cond_feature = self.cond_emb(cond_code)
        else:
            cond_feature = self.cond_emb(cond_feature)
        if self.cond_scale_factor != 1:
            cond_feature = F.interpolate(
                cond_feature.transpose(1, 2), scale_factor=self.cond_scale_factor
            ).transpose(1, 2)
        min_len = min(cond_feature.shape[1], x.shape[1])
        assert x_mask.shape[1] == x.shape[1], "x_mask and x should have the same length"
        x, x_mask, cond_feature = (
            x[:, :min_len],
            x_mask[:, :min_len],
            cond_feature[:, :min_len],
        )

        noise, x, flow_pred, final_mask, prompt_len = self.compute_loss(
            x, x_mask, cond_feature
        )
        return noise, x, flow_pred, final_mask, prompt_len


def voicebox_300M():
    model = VoiceBox(
        mel_dim=128,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        cfg_scale=0.2,
        use_cond_code=True,
        cond_codebook_size=16384,
        cond_dim=1024,
        cond_scale_factor=4,
        sigma=1e-5,
        time_scheduler="cos",
    )
    return model


def extract_normalized_mel_spec_50hz(speech, device="cuda"):
    """
    Extract mel spectrogram from speech.
    Args:
        speech: (B, T)
    Returns:
        mel: (B, T // hop_size, num_mels)
    """
    from dualcodec.utils.melspec import MelSpectrogram

    mel_model = MelSpectrogram(
        sampling_rate=24000,
        n_fft=1920,
        num_mels=128,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=12000,
    ).to(device)
    mel_model.eval()
    mel_feat = mel_model(speech.to(device))
    mel_feat = mel_feat.transpose(1, 2)
    MEL_MIN = -4.92
    MEL_VAR = 8.14
    mel_feat = (mel_feat - MEL_MIN) / math.sqrt(MEL_VAR)
    mel_feat = mel_feat.transpose(1, 2)
    return mel_feat


def mel_spec_50hz():
    from dualcodec.utils.melspec import MelSpectrogram

    mel_model = MelSpectrogram(
        sampling_rate=24000,
        n_fft=1920,
        num_mels=128,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=12000,
    )
    mel_model.eval()
    return mel_model


if __name__ == "__main__":
    model = voicebox_300M()
    size = module_size(model)
    print(size)

    # mel = torch.randn(3, 100, 128)
    # mel_mask = torch.ones(3, 100)
    # cond_code = torch.randint(0, 16384, (3, 25))
    # noise, x, flow_pred, final_mask, prompt_len = model(mel, mel_mask, cond_code)

    # # calculate loss
    # final_mask = final_mask.squeeze(-1)

    # flow_gt = x - (1 - model.sigma) * noise

    # # use l1 loss
    # diff_loss = F.l1_loss(
    #     flow_pred, flow_gt, reduction="none"
    # ).float() * final_mask.unsqueeze(-1)
    # diff_loss = torch.mean(diff_loss, dim=2).sum() / final_mask.sum()
    # print(diff_loss)
