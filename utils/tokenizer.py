# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from
# https://github.com/lifeiteng/vall-e/blob/9c69096d603ce13174fb5cb025f185e2e9b36ac7/valle/data/tokenizer.py

import re
from typing import Any, Dict, List, Optional, Pattern, Union

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio


class AudioTokenizer:
    """EnCodec audio tokenizer for encoding and decoding audio.

    Attributes:
        device: The device on which the codec model is loaded.
        codec: The pretrained EnCodec model.
        sample_rate: Sample rate of the model.
        channels: Number of audio channels in the model.
    """

    def __init__(self, device: Any = None) -> None:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        remove_encodec_weight_norm(model)

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode the audio waveform.

        Args:
            wav: A tensor representing the audio waveform.

        Returns:
            A tensor representing the encoded audio.
        """
        return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        """Decode the encoded audio frames.

        Args:
            frames: A tensor representing the encoded audio frames.

        Returns:
            A tensor representing the decoded audio waveform.
        """
        return self.codec.decode(frames)


def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str):
    """
    Tokenize the audio waveform using the given AudioTokenizer.

    Args:
        tokenizer: An instance of AudioTokenizer.
        audio_path: Path to the audio file.

    Returns:
        A tensor of encoded frames from the audio.

    Raises:
        FileNotFoundError: If the audio file is not found.
        RuntimeError: If there's an error processing the audio data.
    """
    # try:
    # Load and preprocess the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames

    # except FileNotFoundError:
    #     raise FileNotFoundError(f"Audio file not found at {audio_path}")
    # except Exception as e:
    #     raise RuntimeError(f"Error processing audio data: {e}")


def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)


def extract_encodec_token(wav_path):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    wav, sr = torchaudio.load(wav_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    if torch.cuda.is_available():
        model = model.cuda()
        wav = wav.cuda()
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        codes_ = torch.cat(
            [encoded[0] for encoded in encoded_frames], dim=-1
        )  # [B, n_q, T]
        codes = codes_.cpu().numpy()[0, :, :].T  # [T, 8]

        return codes
