# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
import soundfile as sf
import os
from torchaudio.functional import pitch_shift
import librosa
from librosa.filters import mel as librosa_mel_fn
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        super(MelSpectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        mel_basis = {}
        hann_window = {}

        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(win_size)

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("hann_window", hann_window)

    def forward(self, y):
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec)

        return spec


# if __name__ == "__main__":
#     mel_model = MelSpectrogram(
#         sampling_rate=24000,
#         num_mels=128,
#         hop_size=480,
#         n_fft=1920,
#         win_size=1920,
#         fmin=0,
#         fmax=12000,
#     )
#     mel_model = mel_model.to("cuda:0")

#     x_len = 0
#     var = 0
#     mean = 0


#     from utils.util import load_config
#     cfg = load_config("/storage/wyc/SpeechGeneration/egs/tts/SoundStorm/exp_config_16k_emilia_llama_new_semantic_repcodec_8192_1q_24k.json")
#     print(cfg)
#     dataset = SoundStormDataset(AK, SK, bucket_name, cfg=cfg)
#     print(dataset.__getitem__(0))

#     idx_list = list(range(0, len(dataset)))
#     np.random.shuffle(idx_list)
#     for i in tqdm.tqdm(range(10000)):
#         idx = idx_list[i]
#         # data_path = dataset[idx]
#         # speech, _ = librosa.load(data_path, sr=24000)
#         speech = dataset[idx]["speech"]
#         speech = torch.tensor(speech).unsqueeze(0).to("cuda")
#         mel = mel_model(speech)
#         temp_len = mel.shape[-1]
#         temp_mean = mel.mean()
#         temp_var = mel.var()

#         new_mean = (mean * x_len + temp_mean * temp_len) / (x_len + temp_len)
#         new_var = (var * (x_len - 1) + temp_var * (temp_len - 1) + x_len * (new_mean - mean)**2 + temp_len * (new_mean - temp_mean)**2) / (x_len + temp_len -1)

#         x_len += temp_len
#         mean = new_mean
#         var = new_var

#         if i % 100 == 0:
#             print(mean, var)

#     print(mean)
#     print(var)
