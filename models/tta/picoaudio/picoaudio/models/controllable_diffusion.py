import random
import numpy as np
from tqdm import tqdm
from einops import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTokenizer, AutoTokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel
import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers import AutoencoderKL as DiffuserAutoencoderKL

from utils.torch_tools import wav_to_fbank
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder.autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata


def build_pretrained_models(name):
    checkpoint = torch.load(get_metadata()[name]["path"], map_location="cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    vae_state_dict = {
        k[18:]: v
        for k, v in checkpoint["state_dict"].items()
        if "first_stage_model." in k
    }

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    vae.eval()
    fn_STFT.eval()

    return vae, fn_STFT


def _init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


class BaseDiffusion(nn.Module):
    def __init__(
        self,
        scheduler_name,
        unet_model_config_path=None,
        snr_gamma=None,
        uncondition=False,
    ):
        super().__init__()

        assert (
            unet_model_config_path is not None
        ), "Either UNet pretrain model name or a config file path is required"
        self.scheduler_name = scheduler_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.uncondition = uncondition
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.scheduler_name, subfolder="scheduler"
        )
        self.inference_scheduler = DDPMScheduler.from_pretrained(
            self.scheduler_name, subfolder="scheduler"
        )
        unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
        self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")
        print("UNet initialized randomly.")
        """
        self.text_encoder_name = "./checkpoint/models--google--flan-t5-large/" + \
           "snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a/"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        """

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_text(self, input_dict):
        raise NotImplementedError

    def forward(self, input_dict):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, input_dict):
        raise NotImplementedError


class Text_Onset_2_Audio_Diffusion(BaseDiffusion):
    def __init__(
        self,
        scheduler_name,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
    ):
        super().__init__(scheduler_name, unet_model_config_path, snr_gamma, uncondition)
        self.freeze_text_encoder = freeze_text_encoder
        self.class_emb = nn.Embedding(24, 1024)
        # self.channel_emb = nn.Linear(24, 16)
        # _init_layer(self.channel_emb)

    def encode_channel(self, input):
        # input [batch, 32, 256] -> [batch, 2, 256, 16]
        return input.reshape(input.shape[0], 2, 16, 256).transpose(2, 3)
        # return self.channel_emb(input).unsqueeze(1)

    def encode_text(self, input_dict):
        device = self.device

        encoder_hidden_states = self.class_emb(input_dict["event_info"].unsqueeze(-1))
        boolean_encoder_mask = (torch.ones(len(encoder_hidden_states), 1) == 1).to(
            device
        )

        return encoder_hidden_states, boolean_encoder_mask

    def forward(self, input_dict, validation_mode=False):
        device = self.device
        latents = input_dict["latent"]
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        # [batch, 1, 1024], [batch, 1]

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(input_dict)
        if self.uncondition:
            mask_indices = [k for k in range(len(latents)) if random.random() < 0.1]
            if len(mask_indices) > 0:
                encoder_hidden_states[mask_indices] = 0

        bsz = latents.shape[0]
        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps // 2) * torch.ones(
                (bsz,), dtype=torch.int64, device=device
            )
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(
                0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device
            )
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        onset_emb = self.encode_channel(input_dict["onset"])
        # [batch, channel:8, 256, 16] + [batch, onset:2, 256, 16]
        onset_noisy_latents = torch.cat((onset_emb, noisy_latents), dim=1)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        model_pred = self.unet(
            onset_noisy_latents,
            timesteps,
            encoder_hidden_states,
            # encoder_attention_mask=boolean_encoder_mask
        ).sample

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack(
                    [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    def prepare_latents(
        self, batch_size, inference_scheduler, num_channels_latents, dtype, device
    ):
        shape = (batch_size, num_channels_latents, 256, 16)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, input_dict, num_samples_per_prompt):
        device = self.device
        prompt_embeds, boolean_prompt_mask = self.encode_text(input_dict)
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = boolean_prompt_mask.repeat_interleave(
            num_samples_per_prompt, 0
        )
        # get unconditional embeddings for classifier free guidance
        negative_prompt_embeds = torch.zeros(prompt_embeds.shape).to(device)
        uncond_attention_mask = (torch.ones(attention_mask.shape) == 1).to(device)
        # negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        # uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask

    @torch.no_grad()
    def inference(
        self,
        input_dict,
        inference_scheduler,
        num_steps=20,
        guidance_scale=3,
        num_samples_per_prompt=1,
        disable_progress=True,
    ):
        prompt = input_dict["onset"]
        device = self.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) * num_samples_per_prompt

        if classifier_free_guidance:
            prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(
                input_dict, num_samples_per_prompt
            )
        else:
            prompt_embeds, boolean_prompt_mask = self.encode_text(input_dict)
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(
                num_samples_per_prompt, 0
            )

        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels - 2
        latents = self.prepare_latents(
            batch_size,
            inference_scheduler,
            num_channels_latents,
            prompt_embeds.dtype,
            device,
        )
        onset_emb = self.encode_channel(input_dict["onset"]).repeat_interleave(
            num_samples_per_prompt, 0
        )
        onset_latents = torch.cat((onset_emb, latents), dim=1)

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, t in tqdm(enumerate(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([onset_latents] * 2)
                if classifier_free_guidance
                else onset_latents
            )
            latent_model_input = inference_scheduler.scale_model_input(
                latent_model_input, t
            )
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=boolean_prompt_mask,
            ).sample

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample
            onset_latents = torch.cat((onset_emb, latents), dim=1)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0
            ):
                progress_bar.update(1)

        return latents


class ClapText_Onset_2_Audio_Diffusion(Text_Onset_2_Audio_Diffusion):
    def encode_text(self, input_dict):
        device = self.device

        encoder_hidden_states = (
            input_dict["event_info"].repeat_interleave(2, -1).unsqueeze(1)
        )
        boolean_encoder_mask = (torch.ones(len(encoder_hidden_states), 1) == 1).to(
            device
        )

        return encoder_hidden_states, boolean_encoder_mask
