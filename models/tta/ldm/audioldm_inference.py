# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import json

from models.tta.autoencoder.autoencoder import AutoencoderKL
from models.tta.ldm.inference_utils.vocoder import Generator
from models.tta.ldm.audioldm import AudioLDM
from transformers import T5EncoderModel, AutoTokenizer
from diffusers import PNDMScheduler

import matplotlib.pyplot as plt
from scipy.io.wavfile import write


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AudioLDMInference:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args

        self.build_autoencoderkl()
        self.build_textencoder()

        self.model = self.build_model()
        self.load_state_dict()

        self.build_vocoder()

        self.out_path = self.args.output_dir
        self.out_mel_path = os.path.join(self.out_path, "mel")
        self.out_wav_path = os.path.join(self.out_path, "wav")
        os.makedirs(self.out_mel_path, exist_ok=True)
        os.makedirs(self.out_wav_path, exist_ok=True)

    def build_autoencoderkl(self):
        self.autoencoderkl = AutoencoderKL(self.cfg.model.autoencoderkl)
        self.autoencoder_path = self.cfg.model.autoencoder_path
        checkpoint = torch.load(self.autoencoder_path, map_location="cpu")
        self.autoencoderkl.load_state_dict(checkpoint["model"])
        self.autoencoderkl.cuda(self.args.local_rank)
        self.autoencoderkl.requires_grad_(requires_grad=False)
        self.autoencoderkl.eval()

    def build_textencoder(self):
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.text_encoder.cuda(self.args.local_rank)
        self.text_encoder.requires_grad_(requires_grad=False)
        self.text_encoder.eval()

    def build_vocoder(self):
        config_file = os.path.join(self.args.vocoder_config_path)
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        self.vocoder = Generator(h).to(self.args.local_rank)
        checkpoint_dict = torch.load(
            self.args.vocoder_path, map_location=self.args.local_rank
        )
        self.vocoder.load_state_dict(checkpoint_dict["generator"])

    def build_model(self):
        self.model = AudioLDM(self.cfg.model.audioldm)
        return self.model

    def load_state_dict(self):
        self.checkpoint_path = self.args.checkpoint_path
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.cuda(self.args.local_rank)

    def get_text_embedding(self):
        text = self.args.text

        prompt = [text]

        text_input = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            padding="do_not_pad",
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.args.local_rank)
        )[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.args.local_rank)
        )[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def inference(self):
        text_embeddings = self.get_text_embedding()
        print(text_embeddings.shape)

        num_steps = self.args.num_steps
        guidance_scale = self.args.guidance_scale

        noise_scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )

        noise_scheduler.set_timesteps(num_steps)

        latents = torch.randn(
            (
                1,
                self.cfg.model.autoencoderkl.z_channels,
                80 // (2 ** (len(self.cfg.model.autoencoderkl.ch_mult) - 1)),
                624 // (2 ** (len(self.cfg.model.autoencoderkl.ch_mult) - 1)),
            )
        ).to(self.args.local_rank)

        self.model.eval()
        for t in tqdm(noise_scheduler.timesteps):
            t = t.to(self.args.local_rank)

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )
            # print(latent_model_input.shape)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.model(
                    latent_model_input, torch.cat([t.unsqueeze(0)] * 2), text_embeddings
                )

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            # print(latents.shape)

        latents_out = latents
        print(latents_out.shape)

        with torch.no_grad():
            mel_out = self.autoencoderkl.decode(latents_out)
        print(mel_out.shape)

        melspec = mel_out[0, 0].cpu().detach().numpy()
        plt.imsave(os.path.join(self.out_mel_path, self.args.text + ".png"), melspec)

        self.vocoder.eval()
        self.vocoder.remove_weight_norm()
        with torch.no_grad():
            melspec = np.expand_dims(melspec, 0)
            melspec = torch.FloatTensor(melspec).to(self.args.local_rank)

            y = self.vocoder(melspec)
            audio = y.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype("int16")

        write(os.path.join(self.out_wav_path, self.args.text + ".wav"), 16000, audio)
