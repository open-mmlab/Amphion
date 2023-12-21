# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

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
        mels = mels.to(device)
        if f0s != None:
            f0s = f0s.to(device)

        if f0s == None and not cfg.preprocess.extract_amplitude_phase:
            output = model.forward(mels)
        elif cfg.preprocess.extract_amplitude_phase:
            (
                _,
                _,
                _,
                _,
                output,
            ) = model.forward(mels)
        else:
            output = model.forward(mels, f0s)

        return output.squeeze(1).detach().cpu()


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
                audio_length = frame * model.cfg.preprocess.hop_size
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
                audio_length = frame * model.cfg.preprocess.hop_size
                audio = audio[:audio_length]

                audios.append(audio)
    return audios
