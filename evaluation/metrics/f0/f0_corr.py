# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import librosa
import numpy as np
import parselmouth

from utils.f0 import interpolate


def get_cents(f0_hz):
    """
    F_{cent} = 1200 * log2 (F/440)

    Reference:
        APSIPA'17, Perceptual Evaluation of Singing Quality
    """
    voiced_f0 = f0_hz[f0_hz != 0]
    return 1200 * np.log2(voiced_f0 / 440)


def get_pitch_sub_median(f0_hz):
    """
    f0_hz: (,T)
    """
    f0_cent = get_cents(f0_hz)
    return f0_cent - np.median(f0_cent)


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


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def extract_f0_hz(
    wav_path,
    fs=16000,
    hop_length=256,
    f0_min=50,
    f0_max=1100,
):
    cfg = JsonHParams()
    cfg.sample_rate = fs
    cfg.hop_size = hop_length
    cfg.f0_min = f0_min
    cfg.f0_max = f0_max
    cfg.pitch_bin = 256
    cfg.pitch_max = f0_max
    cfg.pitch_min = f0_min

    # Compute f0
    audio, _ = librosa.load(wav_path, sr=fs)
    f0 = get_f0_features_using_parselmouth(
        audio,
        cfg,
    )
    f0, _ = interpolate(f0)
    return f0


def extract_fpc(
    audio_ref,
    audio_deg,
    fs=16000,
    need_mean=True,
    hop_length=256,
    f0_min=50,
    f0_max=1100,
    method="dtw",
):
    """Compute F0 Pearson Distance (FPC) between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    hop_length: hop length.
    f0_min: lower limit for f0.
    f0_max: upper limit for f0.
    pitch_bin: number of bins for f0 quantization.
    pitch_max: upper limit for f0 quantization.
    pitch_min: lower limit for f0 quantization.
    need_mean: subtract the mean value from f0 if "True".
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    """
    # Initialize method
    from torchmetrics import PearsonCorrCoef

    pearson = PearsonCorrCoef()

    # Load audio
    if fs != None:
        audio_ref, _ = librosa.load(audio_ref, sr=fs)
        audio_deg, _ = librosa.load(audio_deg, sr=fs)
    else:
        audio_ref, ref_fs = librosa.load(audio_ref)
        audio_deg, deg_fs = librosa.load(audio_deg)
        assert ref_fs == deg_fs
        fs = ref_fs

    # Initialize config
    cfg = JsonHParams()
    cfg.sample_rate = fs
    cfg.hop_size = hop_length
    cfg.f0_min = f0_min
    cfg.f0_max = f0_max
    cfg.pitch_bin = 256
    cfg.pitch_max = f0_max
    cfg.pitch_min = f0_min

    # Compute f0
    f0_ref = get_f0_features_using_parselmouth(
        audio_ref,
        cfg,
    )

    f0_deg = get_f0_features_using_parselmouth(
        audio_deg,
        cfg,
    )

    # Subtract mean value from f0
    if need_mean:
        f0_ref = torch.from_numpy(f0_ref)
        f0_deg = torch.from_numpy(f0_deg)

        f0_ref = get_pitch_sub_median(f0_ref).numpy()
        f0_deg = get_pitch_sub_median(f0_deg).numpy()

    # Avoid silence
    min_length = min(len(f0_ref), len(f0_deg))
    if min_length <= 1:
        return 1

    # F0 length alignment
    if method == "cut":
        length = min(len(f0_ref), len(f0_deg))
        f0_ref = f0_ref[:length]
        f0_deg = f0_deg[:length]
    elif method == "dtw":
        _, wp = librosa.sequence.dtw(f0_ref, f0_deg, backtrack=True)
        f0_gt_new = []
        f0_pred_new = []
        for i in range(wp.shape[0]):
            gt_index = wp[i][0]
            pred_index = wp[i][1]
            f0_gt_new.append(f0_ref[gt_index])
            f0_pred_new.append(f0_deg[pred_index])
        f0_ref = np.array(f0_gt_new)
        f0_deg = np.array(f0_pred_new)
        assert len(f0_ref) == len(f0_deg)

    # Convert to tensor
    f0_ref = torch.from_numpy(f0_ref)
    f0_deg = torch.from_numpy(f0_deg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        f0_ref = f0_ref.to(device)
        f0_deg = f0_deg.to(device)
        pearson = pearson.to(device)

    return pearson(f0_ref, f0_deg).detach().cpu().numpy().tolist()
