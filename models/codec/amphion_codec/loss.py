import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram
from einops import rearrange
from typing import List


def stft(x, fft_size, hop_size, win_length, window, use_complex=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
    )

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(
            torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-7, max=1e3)
        ).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3)  # [B, 2, T, F]
        return res


def compute_mag_scale(n_fft, sampling_rate):
    frequencies = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    frequencies = np.where(frequencies > 1e-10, frequencies, -10)
    db_scale = librosa.frequency_weighting(frequencies).reshape(1, 1, -1)
    mag_scale = np.sqrt(librosa.db_to_power(db_scale)).astype(np.float32)
    return torch.from_numpy(mag_scale)


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initialize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of ground-truth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag) / torch.norm(y_mag)


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of ground-truth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size=1024,
        hop_length=120,
        win_length=600,
        sampling_rate=16000,
        window="hann_window",
        cfg=None,
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()

        fft_size = (
            cfg.fft_size if cfg is not None and hasattr(cfg, "fft_size") else fft_size
        )
        hop_length = (
            cfg.hop_length
            if cfg is not None and hasattr(cfg, "hop_length")
            else hop_length
        )
        win_length = (
            cfg.win_length
            if cfg is not None and hasattr(cfg, "win_length")
            else win_length
        )
        window = cfg.window if cfg is not None and hasattr(cfg, "window") else window

        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

        self.register_buffer("mag_scale", compute_mag_scale(fft_size, sampling_rate))

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = (
            stft(x, self.fft_size, self.hop_length, self.win_length, self.window)
            * self.mag_scale
        )
        y_mag = (
            stft(y, self.fft_size, self.hop_length, self.win_length, self.window)
            * self.mag_scale
        )
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        log_mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=(1024, 2048, 512),
        hop_sizes=(120, 240, 50),
        win_lengths=(600, 1200, 240),
        window="hann_window",
        sampling_rate=16000,
        cfg=None,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()

        fft_sizes = (
            cfg.fft_sizes
            if cfg is not None and hasattr(cfg, "fft_sizes")
            else fft_sizes
        )
        hop_sizes = (
            cfg.hop_sizes
            if cfg is not None and hasattr(cfg, "hop_sizes")
            else hop_sizes
        )
        win_lengths = (
            cfg.win_lengths
            if cfg is not None and hasattr(cfg, "win_lengths")
            else win_lengths
        )
        window = cfg.window if cfg is not None and hasattr(cfg, "window") else window
        sampling_rate = (
            cfg.sampling_rate
            if cfg is not None and hasattr(cfg, "sampling_rate")
            else sampling_rate
        )

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(fs, ss, wl, window=window, sampling_rate=sampling_rate)
            ]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): GroundTruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class MultiResolutionMelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        mel_fmin: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        cfg=None,
    ):
        super().__init__()

        sample_rate = (
            cfg.sample_rate
            if cfg is not None and hasattr(cfg, "sample_rate")
            else sample_rate
        )
        n_mels = cfg.n_mels if cfg is not None and hasattr(cfg, "n_mels") else n_mels
        window_lengths = (
            cfg.window_lengths
            if cfg is not None and hasattr(cfg, "window_lengths")
            else window_lengths
        )
        clamp_eps = (
            cfg.clamp_eps
            if cfg is not None and hasattr(cfg, "clamp_eps")
            else clamp_eps
        )
        mag_weight = (
            cfg.mag_weight
            if cfg is not None and hasattr(cfg, "mag_weight")
            else mag_weight
        )
        log_weight = (
            cfg.log_weight
            if cfg is not None and hasattr(cfg, "log_weight")
            else log_weight
        )
        pow = cfg.pow if cfg is not None and hasattr(cfg, "pow") else pow
        mel_fmin = (
            cfg.mel_fmin if cfg is not None and hasattr(cfg, "mel_fmin") else mel_fmin
        )
        mel_fmax = (
            cfg.mel_fmax if cfg is not None and hasattr(cfg, "mel_fmax") else mel_fmax
        )

        self.mel_transforms = nn.ModuleList(
            [
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    n_mels=n_mel,
                    power=1.0,
                    center=True,
                    norm="slaney",
                    mel_scale="slaney",
                )
                for n_mel, window_length in zip(n_mels, window_lengths)
            ]
        )
        self.n_mels = n_mels
        self.loss_fn = nn.L1Loss()
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def delta(self, x, k):
        l = x.shape[1]
        return x[:, 0 : l - k] - x[:, k:l]

    def forward(self, x, y, mask=None):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for mel_transform in self.mel_transforms:
            x_mel = mel_transform(x)
            y_mel = mel_transform(y)
            log_x_mel = x_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            log_y_mel = y_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            loss += self.log_weight * self.loss_fn(log_x_mel, log_y_mel)
            loss += self.mag_weight * self.loss_fn(x_mel, y_mel)
            # loss += self.loss_fn(self.delta(log_x_mel, 1), self.delta(log_y_mel, 1))
            # log_x_mel = rearrange(log_x_mel, 'b c t -> b t c')
            # log_y_mel = rearrange(log_y_mel, 'b c t -> b t c')
            # for i in range(3):
            #     loss += self.loss_fn(self.delta(log_x_mel, i), self.delta(log_y_mel, i))
        # loss /= len(self.mel_transforms)
        return loss


class GANLoss(nn.Module):
    def __init__(self, mode="lsgan"):
        super(GANLoss, self).__init__()
        assert mode in ["lsgan", "lsgan_std", "hinge"]
        self.mode = mode

    def disc_loss(self, real, fake):
        if self.mode == "lsgan":
            real_loss = F.mse_loss(real, torch.ones_like(real))
            fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
        elif self.mode == "lsgan_std":
            real = (real - 1.0).pow(2)
            fake = (fake - 0.0).pow(2)
            real_loss = real.mean() + real.std()
            fake_loss = fake.mean() + fake.std()
        elif self.mode == "hinge":
            real_loss = torch.relu(1.0 - real).mean()
            fake_loss = torch.relu(1.0 + fake).mean()
        else:
            raise ValueError(f"no such mode {self.mode}")

        return real_loss, fake_loss

    def disc_loss2(self, fake):
        if self.mode == "lsgan":
            fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
        elif self.mode == "lsgan_std":
            fake = (fake - 0.0).pow(2)
            fake_loss = fake.mean() + fake.std()
        elif self.mode == "hinge":
            fake_loss = torch.relu(1.0 + fake).mean()
        else:
            raise ValueError(f"no such mode {self.mode}")

        return fake_loss

    def gen_loss(self, fake):
        if self.mode == "lsgan":
            gen_loss = F.mse_loss(fake, torch.ones_like(fake))
        elif self.mode == "lsgan_std":
            fake = (fake - 1.0).pow(2)
            gen_loss = fake.mean() + fake.std()
        elif self.mode == "hinge":
            gen_loss = -fake.mean()
        else:
            raise ValueError(f"no such mode {self.mode}")

        return gen_loss
