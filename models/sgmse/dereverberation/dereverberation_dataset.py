import torch
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import os
from os.path import join


class Specs:
    def __init__(self, cfg, subset, shuffle_spec):
        self.cfg = cfg
        self.data_dir = os.path.join(
            cfg.preprocess.processed_dir, cfg.dataset[0], "audio"
        )
        self.clean_files = sorted(glob(join(self.data_dir, subset) + "/anechoic/*.wav"))
        self.noisy_files = sorted(glob(join(self.data_dir, subset) + "/reverb/*.wav"))
        self.dummy = cfg.preprocess.dummy
        self.num_frames = cfg.preprocess.num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = cfg.preprocess.normalize
        self.hop_length = cfg.preprocess.hop_length
        self.n_fft = cfg.preprocess.n_fft
        self.window = self.get_window(self.n_fft)
        self.windows = {}
        self.spec_abs_exponent = cfg.preprocess.spec_abs_exponent
        self.spec_factor = cfg.preprocess.spec_factor

    def __getitem__(self, i):
        x, _ = load(self.clean_files[i])
        y, _ = load(self.noisy_files[i])

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len - target_len) / 2)
            x = x[..., start : start + target_len]
            y = y[..., start : start + target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode="constant")
            y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode="constant")

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs())
        Y = torch.stft(y, **self.stft_kwargs())
        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return {"X": X, "Y": Y}

    def __len__(self):
        if self.dummy:
            return int(len(self.clean_files) / 200)
        else:
            return len(self.clean_files)

    def spec_transform(self, spec):
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs() ** e * torch.exp(1j * spec.angle())
        spec = spec * self.spec_factor

        return spec

    def stft_kwargs(self):
        return {**self.istft_kwargs(), "return_complex": True}

    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
        )

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs(), "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(
            spec, **{**self.istft_kwargs(), "window": window, "length": length}
        )

    @staticmethod
    def get_window(window_length):
        return torch.hann_window(window_length, periodic=True)

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window
