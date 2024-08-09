# -*- coding: utf-8 -*-

import torch
from models.tts.UniCATS.CTXvec2wav.build_model.models.conformer.decoder import Decoder as ConformerDecoder
from models.tts.UniCATS.CTXvec2wav.utils import crop_seq
from models.tts.UniCATS.CTXvec2wav.build_model.models.hifigan import HiFiGANGenerator


class CTXVEC2WAVFrontend(torch.nn.Module):

    def __init__(self,
                 num_mels,
                 aux_channels,
                 vqvec_channels,
                 prompt_channels,
                 conformer_params):

        super(CTXVEC2WAVFrontend, self).__init__()

        self.prompt_prenet = torch.nn.Conv1d(prompt_channels, conformer_params["attention_dim"], kernel_size=5) # , padding=2)
        self.encoder1 = ConformerDecoder(vqvec_channels, input_layer='linear', **conformer_params)

        self.aux_prenet = torch.nn.Conv1d(aux_channels, conformer_params["attention_dim"], kernel_size=5, padding=2)
        self.aux_proj = torch.nn.Linear(conformer_params["attention_dim"], aux_channels)
        self.hidden_proj = torch.nn.Linear(conformer_params["attention_dim"], conformer_params["attention_dim"])

        self.encoder2 = ConformerDecoder(0, input_layer=None, **conformer_params)
        self.mel_proj = torch.nn.Linear(conformer_params["attention_dim"], num_mels)
        
    def forward(self, vqvec, prompt, mask=None, prompt_mask=None, aux=None):
        """
        params:
            vqvec: sequence of VQ-vectors.
            prompt: sequence of mel-spectrogram prompt (acoustic context)
            mask: mask of the vqvec. True or 1 stands for valid values.
            prompt_mask: mask of the prompt.
            aux: auxiliary features (log probability of voice, log pitch, log energy).
        vqvec and prompt are of shape [B, D, T]. All masks are of shape [B, T].
        returns:
            enc_out: the input to the CTX-vec2wav Generator (HifiGAN);
            mel: the frontend predicted mel spectrogram (for faster convergence);
            aux_pred: the predicted auxiliary features.
        """
        prompt = self.prompt_prenet(prompt.transpose(1, 2)).transpose(1, 2)

        # Note: you can verify that the output is the same if we shuffle the prompt before cross attention (when inference)
        # num_samples = prompt.size(1)
        # permuted_indices = torch.randperm(num_samples)
        # prompt = prompt[:, permuted_indices, :]
        # Note: or flip it
        # prompt = torch.flip(prompt, dims=[1])
        if mask is not None:
            mask = mask.unsqueeze(-2)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.unsqueeze(-2)
        enc_out, _ = self.encoder1(vqvec, mask, prompt, prompt_mask)

        aux_pred = self.aux_proj(enc_out)  # (B, L, 3)
        if aux is None:
            aux = aux_pred
        h = self.hidden_proj(enc_out) + self.aux_prenet(aux.transpose(1, 2)).transpose(1, 2)

        enc_out, _ = self.encoder2(h, mask, prompt, prompt_mask)
        mel = self.mel_proj(enc_out)  # (B, L, 80)

        return enc_out, mel, aux_pred


class CTXVEC2WAVGenerator(torch.nn.Module):

    def __init__(self, frontend: CTXVEC2WAVFrontend, backend: HiFiGANGenerator):

        super(CTXVEC2WAVGenerator, self).__init__()
        self.frontend = frontend
        self.backend = backend

    def forward(self, vqvec, prompt, mask=None, prompt_mask=None, crop_len=0, crop_offsets=None, aux=None):
        """
        :param vqvec: (torch.Tensor) The shape is (B, L, 512). Sequence of VQ-vectors.
        :param prompt: (torch.Tensor) The shape is (B, L', 80). Sequence of mel-spectrogram prompt (acoustic context)
        :param mask: (torch.Tensor) The dtype is torch.bool. The shape is (B, L). True or 1 stands for valid values in `vqvec`.
        :param prompt_mask: (torch.Tensor) The dtype is torch.bool. The shape is (B, L'). True or 1 stands for valid values in `prompt`.
        :return: frontend predicted mel spectrogram; predicted auxiliary features; reconstructed waveform.
        """
        h, mel, aux = self.frontend(vqvec, prompt, mask=mask, prompt_mask=prompt_mask, aux=aux)  # (B, L, adim), (B, L, 80)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0)
        h = h.transpose(1, 2)
        if crop_len > 0:
            h = crop_seq(h, crop_offsets, crop_len)
        wav = self.backend(h)  # (B, C, T)
        return mel, aux, wav

    def inference(self, vqvec, prompt, aux=None, **kwargs):
        # if aux is provided, force the frontend to use the provided aux feature.
        h, mel, aux = self.frontend(vqvec, prompt, aux=aux)
        wav = self.backend.inference(h, **kwargs)

        return mel, aux, wav