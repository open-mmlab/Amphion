# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from tqdm import tqdm
from utils.util import pad_mels_to_tensors, pad_f0_to_tensors


def vocoder_inference(cfg, model, mels, f0s=None, device=None, fast_inference=False):
    """Inference the vocoder
    Args:
        mels: A tensor of mel-specs with the shape (batch_size, num_mels, frames)
    Returns:
        audios: A tensor of audios with the shape (batch_size, seq_len)
    """
    model.eval()

    with torch.no_grad():
        training_noise_schedule = np.array(cfg.model.diffwave.noise_schedule)
        inference_noise_schedule = (
            np.array(cfg.model.diffwave.inference_noise_schedule)
            if fast_inference
            else np.array(cfg.model.diffwave.noise_schedule)
        )

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5
                    )
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        mels = mels.to(device)
        audio = torch.randn(
            mels.shape[0],
            cfg.preprocess.hop_size * mels.shape[-1],
            device=device,
        )

        for n in tqdm(range(len(alpha) - 1, -1, -1)):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            audio = c1 * (
                audio
                - c2
                * model(audio, torch.tensor([T[n]], device=audio.device), mels).squeeze(
                    1
                )
            )
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = (
                    (1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]
                ) ** 0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)

    return audio.detach().cpu()


def synthesis_audios(cfg, model, mels, f0s=None, batch_size=None, fast_inference=False):
    """Inference the vocoder
    Args:
        mels: A list of mel-specs
    Returns:
        audios: A list of audios
    """
    # Get the device
    device = next(model.parameters()).device

    audios = []

    # Pad the given list into tensors
    mel_batches, mel_frames = pad_mels_to_tensors(mels, batch_size)
    if f0s != None:
        f0_batches = pad_f0_to_tensors(f0s, batch_size)

    if f0s == None:
        for mel_batch, mel_frame in zip(mel_batches, mel_frames):
            for i in range(mel_batch.shape[0]):
                mel = mel_batch[i]
                frame = mel_frame[i]
                audio = vocoder_inference(
                    cfg,
                    model,
                    mel.unsqueeze(0),
                    device=device,
                    fast_inference=fast_inference,
                ).squeeze(0)

                # calculate the audio length
                audio_length = frame * cfg.preprocess.hop_size
                audio = audio[:audio_length]

                audios.append(audio)
    else:
        for mel_batch, f0_batch, mel_frame in zip(mel_batches, f0_batches, mel_frames):
            for i in range(mel_batch.shape[0]):
                mel = mel_batch[i]
                f0 = f0_batch[i]
                frame = mel_frame[i]
                audio = vocoder_inference(
                    cfg,
                    model,
                    mel.unsqueeze(0),
                    f0s=f0.unsqueeze(0),
                    device=device,
                    fast_inference=fast_inference,
                ).squeeze(0)

                # calculate the audio length
                audio_length = frame * cfg.preprocess.hop_size
                audio = audio[:audio_length]

                audios.append(audio)
    return audios
