# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import librosa
import numpy as np
import torch
import parselmouth
import torchcrepe
import pyworld as pw


def f0_to_coarse(f0, pitch_bin, f0_min, f0_max):
    """
    Convert f0 (Hz) to pitch (mel scale), and then quantize the mel-scale pitch to the
    range from [1, 2, 3, ..., pitch_bin-1]

    Reference: https://en.wikipedia.org/wiki/Mel_scale

    Args:
        f0 (array or Tensor): Hz
        pitch_bin (int): the vocabulary size
        f0_min (int): the minimum f0 (Hz)
        f0_max (int): the maximum f0 (Hz)

    Returns:
        quantized f0 (array or Tensor)
    """
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (pitch_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > pitch_bin - 1] = pitch_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int32)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse


def interpolate(f0):
    """Interpolate the unvoiced part. Thus the f0 can be passed to a subtractive synthesizer.
    Args:
        f0: A numpy array of shape (seq_len,)
    Returns:
        f0: Interpolated f0 of shape (seq_len,)
        uv: Unvoiced part of shape (seq_len,)
    """
    uv = f0 == 0
    if len(f0[~uv]) > 0:
        # interpolate the unvoiced f0
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        uv = uv.astype("float")
        uv = np.min(np.array([uv[:-2], uv[1:-1], uv[2:]]), axis=0)
        uv = np.pad(uv, (1, 1))
    return f0, uv


def get_log_f0(f0):
    f0[np.where(f0 == 0)] = 1
    log_f0 = np.log(f0)
    return log_f0


def get_f0_features_using_pyin(audio, cfg):
    """Using pyin to extract the f0 feature.
    Args:
        audio
        fs
        win_length
        hop_length
        f0_min
        f0_max
    Returns:
        f0: numpy array of shape (frame_len,)
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=audio,
        fmin=cfg.f0_min,
        fmax=cfg.f0_max,
        sr=cfg.sample_rate,
        win_length=cfg.win_size,
        hop_length=cfg.hop_size,
    )
    # Set nan to 0
    f0[voiced_flag == False] = 0
    return f0


def get_f0_features_using_parselmouth(audio, cfg, speed=1):
    """Using parselmouth to extract the f0 feature.
    Args:
        audio
        mel_len
        hop_length
        fs
        f0_min
        f0_max
        speed(default=1)
    Returns:
        f0: numpy array of shape (frame_len,)
        pitch_coarse: numpy array of shape (frame_len,)
    """
    hop_size = int(np.round(cfg.hop_size * speed))

    # Calculate the time step for pitch extraction
    time_step = hop_size / cfg.sample_rate * 1000

    f0 = (
        parselmouth.Sound(audio, cfg.sample_rate)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=0.6,
            pitch_floor=cfg.f0_min,
            pitch_ceiling=cfg.f0_max,
        )
        .selected_array["frequency"]
    )
    return f0


def get_f0_features_using_dio(audio, cfg):
    """Using dio to extract the f0 feature.
    Args:
        audio
        mel_len
        fs
        hop_length
        f0_min
        f0_max
    Returns:
        f0: numpy array of shape (frame_len,)
    """
    # Get the raw f0
    _f0, t = pw.dio(
        audio.astype("double"),
        cfg.sample_rate,
        f0_floor=cfg.f0_min,
        f0_ceil=cfg.f0_max,
        channels_in_octave=2,
        frame_period=(1000 * cfg.hop_size / cfg.sample_rate),
    )
    # Get the f0
    f0 = pw.stonemask(audio.astype("double"), _f0, t, cfg.sample_rate)
    return f0


def get_f0_features_using_harvest(audio, mel_len, fs, hop_length, f0_min, f0_max):
    """Using harvest to extract the f0 feature.
    Args:
        audio
        mel_len
        fs
        hop_length
        f0_min
        f0_max
    Returns:
        f0: numpy array of shape (frame_len,)
    """
    f0, _ = pw.harvest(
        audio.astype("double"),
        fs,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        frame_period=(1000 * hop_length / fs),
    )
    f0 = f0.astype("float")[:mel_len]
    return f0


def get_f0_features_using_crepe(
    audio, mel_len, fs, hop_length, hop_length_new, f0_min, f0_max, threshold=0.3
):
    """Using torchcrepe to extract the f0 feature.
    Args:
        audio
        mel_len
        fs
        hop_length
        hop_length_new
        f0_min
        f0_max
        threshold(default=0.3)
    Returns:
        f0: numpy array of shape (frame_len,)
    """
    # Currently, crepe only supports 16khz audio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_16k = librosa.resample(audio, orig_sr=fs, target_sr=16000)
    audio_16k_torch = torch.FloatTensor(audio_16k).unsqueeze(0).to(device)

    # Get the raw pitch
    f0, pd = torchcrepe.predict(
        audio_16k_torch,
        16000,
        hop_length_new,
        f0_min,
        f0_max,
        pad=True,
        model="full",
        batch_size=1024,
        device=device,
        return_periodicity=True,
    )

    # Filter, de-silence, set up threshold for unvoiced part
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.0)(pd, audio_16k_torch, 16000, hop_length_new)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    # Convert unvoiced part to 0hz
    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    # Interpolate f0
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(mel_len) * hop_length / fs
    f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    return f0


def get_f0(audio, cfg, use_interpolate=False, return_uv=False):
    if cfg.pitch_extractor == "dio":
        f0 = get_f0_features_using_dio(audio, cfg)
    elif cfg.pitch_extractor == "pyin":
        f0 = get_f0_features_using_pyin(audio, cfg)
    elif cfg.pitch_extractor == "parselmouth":
        f0 = get_f0_features_using_parselmouth(audio, cfg)

    if use_interpolate:
        f0, uv = interpolate(f0)
    else:
        uv = f0 == 0

    if return_uv:
        return f0, uv

    return f0


def get_cents(f0_hz):
    """
    F_{cent} = 1200 * log2 (F/440)

    Reference:
        APSIPA'17, Perceptual Evaluation of Singing Quality
    """
    voiced_f0 = f0_hz[f0_hz != 0]
    return 1200 * np.log2(voiced_f0 / 440)


def get_pitch_derivatives(f0_hz):
    """
    f0_hz: (,T)
    """
    f0_cent = get_cents(f0_hz)
    return f0_cent[1:] - f0_cent[:-1]


def get_pitch_sub_median(f0_hz):
    """
    f0_hz: (,T)
    """
    f0_cent = get_cents(f0_hz)
    return f0_cent - np.median(f0_cent)
