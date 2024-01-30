# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def extract_linear_features(y, cfg, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    hann_window[str(y.device)] = torch.hann_window(cfg.win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((cfg.n_fft - cfg.hop_size) / 2), int((cfg.n_fft - cfg.hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.squeeze(spec, 0)
    return spec


def mel_spectrogram_torch(y, cfg, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if cfg.fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mel,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )
        mel_basis[str(cfg.fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(cfg.win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((cfg.n_fft - cfg.hop_size) / 2), int((cfg.n_fft - cfg.hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[str(cfg.fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


mel_basis = {}
hann_window = {}


def extract_mel_features(
    y,
    cfg,
    center=False,
    # n_fft, n_mel, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """Extract mel features

    Args:
        y (tensor): audio data in tensor
        cfg (dict): configuration in cfg.preprocess
        center (bool, optional): In STFT, whether t-th frame is centered at time t*hop_length. Defaults to False.

    Returns:
        tensor: a tensor containing the mel feature calculated based on STFT result
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if cfg.fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mel,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )
        mel_basis[str(cfg.fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(cfg.win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((cfg.n_fft - cfg.hop_size) / 2), int((cfg.n_fft - cfg.hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(cfg.fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec.squeeze(0)


def extract_mel_features_tts(
    y,
    cfg,
    center=False,
    taco=False,
    _stft=None,
):
    """Extract mel features

    Args:
        y (tensor): audio data in tensor
        cfg (dict): configuration in cfg.preprocess
        center (bool, optional): In STFT, whether t-th frame is centered at time t*hop_length. Defaults to False.
        taco: use tacotron mel

    Returns:
        tensor: a tensor containing the mel feature calculated based on STFT result
    """
    if not taco:
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

        global mel_basis, hann_window
        if cfg.fmax not in mel_basis:
            mel = librosa_mel_fn(
                sr=cfg.sample_rate,
                n_fft=cfg.n_fft,
                n_mels=cfg.n_mel,
                fmin=cfg.fmin,
                fmax=cfg.fmax,
            )
            mel_basis[str(cfg.fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            hann_window[str(y.device)] = torch.hann_window(cfg.win_size).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((cfg.n_fft - cfg.hop_size) / 2), int((cfg.n_fft - cfg.hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        # complex tensor as default, then use view_as_real for future pytorch compatibility
        spec = torch.stft(
            y,
            cfg.n_fft,
            hop_length=cfg.hop_size,
            win_length=cfg.win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(mel_basis[str(cfg.fmax) + "_" + str(y.device)], spec)
        spec = spectral_normalize_torch(spec)
    else:
        audio = torch.clip(y, -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        spec, energy = _stft.mel_spectrogram(audio)

    return spec.squeeze(0)


def amplitude_phase_spectrum(y, cfg):
    hann_window = torch.hann_window(cfg.win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((cfg.n_fft - cfg.hop_size) / 2), int((cfg.n_fft - cfg.hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    stft_spec = torch.stft(
        y,
        cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        window=hann_window,
        center=False,
        return_complex=True,
    )

    stft_spec = torch.view_as_real(stft_spec)
    if stft_spec.size()[0] == 1:
        stft_spec = stft_spec.squeeze(0)

    if len(list(stft_spec.size())) == 4:
        rea = stft_spec[:, :, :, 0]  # [batch_size, n_fft//2+1, frames]
        imag = stft_spec[:, :, :, 1]  # [batch_size, n_fft//2+1, frames]
    else:
        rea = stft_spec[:, :, 0]  # [n_fft//2+1, frames]
        imag = stft_spec[:, :, 1]  # [n_fft//2+1, frames]

    log_amplitude = torch.log(
        torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(imag, 2))) + 1e-5
    )  # [n_fft//2+1, frames]
    phase = torch.atan2(imag, rea)  # [n_fft//2+1, frames]

    return log_amplitude, phase, rea, imag
