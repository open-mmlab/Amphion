# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import os
import json

import numpy as np
import parselmouth
import torch
import torchaudio
from tqdm import tqdm

from audiomentations import TimeStretch

from pedalboard import (
    Pedalboard,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
    PitchShift,
)

from utils.util import has_existed

PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT = 0.0
PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT = 1.0


def wav_to_Sound(wav, sr: int) -> parselmouth.Sound:
    """Convert a waveform to a parselmouth.Sound object

    Args:
        wav (np.ndarray/torch.Tensor): waveform of shape (n_channels, n_samples)
        sr (int, optional): sampling rate.

    Returns:
        parselmouth.Sound: a parselmouth.Sound object
    """
    assert wav.shape == (1, len(wav[0])), "wav must be of shape (1, n_samples)"
    sound = None
    if isinstance(wav, np.ndarray):
        sound = parselmouth.Sound(wav[0], sampling_frequency=sr)
    elif isinstance(wav, torch.Tensor):
        sound = parselmouth.Sound(wav[0].numpy(), sampling_frequency=sr)
    assert sound is not None, "wav must be either np.ndarray or torch.Tensor"
    return sound


def get_pitch_median(wav, sr: int):
    """Get the median pitch of a waveform

    Args:
        wav (np.ndarray/torch.Tensor): waveform of shape (n_channels, n_samples)
        sr (int, optional): sampling rate.

    Returns:
        parselmouth.Pitch, float: a parselmouth.Pitch object and the median pitch
    """
    if not isinstance(wav, parselmouth.Sound):
        sound = wav_to_Sound(wav, sr)
    else:
        sound = wav
    pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT

    # To Pitch: Time step(s)(standard value: 0.0), Pitch floor (Hz)(standard value: 75), Pitch ceiling (Hz)(standard value: 600.0)
    pitch = parselmouth.praat.call(sound, "To Pitch", 0.8 / 75, 75, 600)
    # Get quantile: From time (s), To time (s), Quantile(0.5 is then the 50% quantile, i.e., the median), Units (Hertz or Bark)
    pitch_median = parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")

    return pitch, pitch_median


def change_gender(
    sound,
    pitch=None,
    formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
    new_pitch_median: float = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT,
    pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
    duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT,
) -> parselmouth.Sound:
    """Invoke change gender function in praat

    Args:
        sound (parselmouth.Sound): a parselmouth.Sound object
        pitch (parselmouth.Pitch, optional): a parselmouth.Pitch object. Defaults to None.
        formant_shift_ratio (float, optional): formant shift ratio. A value of 1.0 means no change. Greater than 1.0 means higher pitch. Less than 1.0 means lower pitch.
        new_pitch_median (float, optional): new pitch median.
        pitch_range_ratio (float, optional): pitch range ratio. A value of 1.0 means no change. Greater than 1.0 means higher pitch range. Less than 1.0 means lower pitch range.
        duration_factor (float, optional): duration factor. A value of 1.0 means no change. Greater than 1.0 means longer duration. Less than 1.0 means shorter duration.

    Returns:
        parselmouth.Sound: a parselmouth.Sound object
    """
    if pitch is None:
        new_sound = parselmouth.praat.call(
            sound,
            "Change gender",
            75,
            600,
            formant_shift_ratio,
            new_pitch_median,
            pitch_range_ratio,
            duration_factor,
        )
    else:
        new_sound = parselmouth.praat.call(
            (sound, pitch),
            "Change gender",
            formant_shift_ratio,
            new_pitch_median,
            pitch_range_ratio,
            duration_factor,
        )
    return new_sound


def apply_formant_and_pitch_shift(
    sound: parselmouth.Sound,
    formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
    pitch_shift_ratio: float = PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT,
    pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
    duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT,
) -> parselmouth.Sound:
    """use Praat "Changer gender" command to manipulate pitch and formant
    "Change gender": Praat -> Sound Object -> Convert -> Change gender
    refer to Help of Praat for more details
    # https://github.com/YannickJadoul/Parselmouth/issues/25#issuecomment-608632887 might help
    """
    pitch = None
    new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
    if pitch_shift_ratio != 1.0:
        pitch, pitch_median = get_pitch_median(sound, sound.sampling_frequency)
        new_pitch_median = pitch_median * pitch_shift_ratio

        # refer to https://github.com/praat/praat/issues/1926#issuecomment-974909408
        pitch_minimum = parselmouth.praat.call(
            pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic"
        )
        new_median = pitch_median * pitch_shift_ratio
        scaled_minimum = pitch_minimum * pitch_shift_ratio
        result_minimum = new_median + (scaled_minimum - new_median) * pitch_range_ratio
        if result_minimum < 0:
            new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
            pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

        if math.isnan(new_pitch_median):
            new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
            pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

    new_sound = change_gender(
        sound,
        pitch,
        formant_shift_ratio,
        new_pitch_median,
        pitch_range_ratio,
        duration_factor,
    )
    return new_sound


# Function used in EQ
def pedalboard_equalizer(wav: np.ndarray, sr: int) -> np.ndarray:
    """Use pedalboard to do equalizer"""
    board = Pedalboard()

    cutoff_low_freq = 60
    cutoff_high_freq = 10000

    q_min = 2
    q_max = 5

    random_all_freq = True
    num_filters = 10
    if random_all_freq:
        key_freqs = [random.uniform(1, 12000) for _ in range(num_filters)]
    else:
        key_freqs = [
            power_ratio(float(z) / (num_filters - 1), cutoff_low_freq, cutoff_high_freq)
            for z in range(num_filters)
        ]
    q_values = [
        power_ratio(random.uniform(0, 1), q_min, q_max) for _ in range(num_filters)
    ]
    gains = [random.uniform(-12, 12) for _ in range(num_filters)]
    # low-shelving filter
    board.append(
        LowShelfFilter(
            cutoff_frequency_hz=key_freqs[0], gain_db=gains[0], q=q_values[0]
        )
    )
    # peaking filters
    for i in range(1, 9):
        board.append(
            PeakFilter(
                cutoff_frequency_hz=key_freqs[i], gain_db=gains[i], q=q_values[i]
            )
        )
    # high-shelving filter
    board.append(
        HighShelfFilter(
            cutoff_frequency_hz=key_freqs[9], gain_db=gains[9], q=q_values[9]
        )
    )

    # Apply the pedalboard to the audio
    processed_audio = board(wav, sr)
    return processed_audio


def power_ratio(r: float, a: float, b: float):
    return a * math.pow((b / a), r)


def audiomentations_time_stretch(wav: np.ndarray, sr: int) -> np.ndarray:
    """Use audiomentations to do time stretch"""
    transform = TimeStretch(
        min_rate=0.8, max_rate=1.25, leave_length_unchanged=False, p=1.0
    )
    augmented_wav = transform(wav, sample_rate=sr)
    return augmented_wav


def formant_and_pitch_shift(
    sound: parselmouth.Sound, fs: bool, ps: bool
) -> parselmouth.Sound:
    """ """
    formant_shift_ratio = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT
    pitch_shift_ratio = PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT
    pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

    assert fs != ps, "fs, ps are mutually exclusive"

    if fs:
        formant_shift_ratio = random.uniform(1.0, 1.4)
        use_reciprocal = random.uniform(-1, 1) > 0
        if use_reciprocal:
            formant_shift_ratio = 1.0 / formant_shift_ratio
        # only use praat to change formant
        new_sound = apply_formant_and_pitch_shift(
            sound,
            formant_shift_ratio=formant_shift_ratio,
        )
        return new_sound

    if ps:
        board = Pedalboard()
        board.append(PitchShift(random.uniform(-12, 12)))
        wav_numpy = sound.values
        wav_numpy = board(wav_numpy, sound.sampling_frequency)
        # use pedalboard to change pitch
        new_sound = parselmouth.Sound(
            wav_numpy, sampling_frequency=sound.sampling_frequency
        )
        return new_sound


def wav_manipulation(
    wav: torch.Tensor,
    sr: int,
    aug_type: str = "None",
    formant_shift: bool = False,
    pitch_shift: bool = False,
    time_stretch: bool = False,
    equalizer: bool = False,
) -> torch.Tensor:
    assert aug_type == "None" or aug_type in [
        "formant_shift",
        "pitch_shift",
        "time_stretch",
        "equalizer",
    ], "aug_type must be one of formant_shift, pitch_shift, time_stretch, equalizer"

    assert aug_type == "None" or (
        formant_shift == False
        and pitch_shift == False
        and time_stretch == False
        and equalizer == False
    ), "if aug_type is specified, other argument must be False"

    if aug_type != "None":
        if aug_type == "formant_shift":
            formant_shift = True
        if aug_type == "pitch_shift":
            pitch_shift = True
        if aug_type == "equalizer":
            equalizer = True
        if aug_type == "time_stretch":
            time_stretch = True

    wav_numpy = wav.numpy()

    if equalizer:
        wav_numpy = pedalboard_equalizer(wav_numpy, sr)

    if time_stretch:
        wav_numpy = audiomentations_time_stretch(wav_numpy, sr)

    sound = wav_to_Sound(wav_numpy, sr)

    if formant_shift or pitch_shift:
        sound = formant_and_pitch_shift(sound, formant_shift, pitch_shift)

    wav = torch.from_numpy(sound.values).float()
    # shape (1, n_samples)
    return wav


def augment_dataset(cfg, dataset) -> list:
    """Augment dataset with formant_shift, pitch_shift, time_stretch, equalizer

    Args:
        cfg (dict): configuration
        dataset (str): dataset name

    Returns:
        list: augmented dataset names
    """
    # load metadata
    dataset_path = os.path.join(cfg.preprocess.processed_dir, dataset)
    split = ["train", "test"] if "eval" not in dataset else ["test"]
    augment_datasets = []
    aug_types = [
        "formant_shift" if cfg.preprocess.use_formant_shift else None,
        "pitch_shift" if cfg.preprocess.use_pitch_shift else None,
        "time_stretch" if cfg.preprocess.use_time_stretch else None,
        "equalizer" if cfg.preprocess.use_equalizer else None,
    ]
    aug_types = filter(None, aug_types)
    for aug_type in aug_types:
        print("Augmenting {} with {}...".format(dataset, aug_type))
        new_dataset = dataset + "_" + aug_type
        augment_datasets.append(new_dataset)
        new_dataset_path = os.path.join(cfg.preprocess.processed_dir, new_dataset)

        for dataset_type in split:
            metadata_path = os.path.join(dataset_path, "{}.json".format(dataset_type))
            augmented_metadata = []
            new_metadata_path = os.path.join(
                new_dataset_path, "{}.json".format(dataset_type)
            )
            os.makedirs(new_dataset_path, exist_ok=True)
            new_dataset_wav_dir = os.path.join(new_dataset_path, "wav")
            os.makedirs(new_dataset_wav_dir, exist_ok=True)

            if has_existed(new_metadata_path):
                continue

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            for utt in tqdm(metadata):
                original_wav_path = utt["Path"]
                original_wav, sr = torchaudio.load(original_wav_path)
                new_wav = wav_manipulation(original_wav, sr, aug_type=aug_type)
                new_wav_path = os.path.join(new_dataset_wav_dir, utt["Uid"] + ".wav")
                torchaudio.save(new_wav_path, new_wav, sr)
                new_utt = {
                    "Dataset": utt["Dataset"] + "_" + aug_type,
                    "index": utt["index"],
                    "Singer": utt["Singer"],
                    "Uid": utt["Uid"],
                    "Path": new_wav_path,
                    "Duration": utt["Duration"],
                }
                augmented_metadata.append(new_utt)
            new_metadata_path = os.path.join(
                new_dataset_path, "{}.json".format(dataset_type)
            )
            with open(new_metadata_path, "w") as f:
                json.dump(augmented_metadata, f, indent=4, ensure_ascii=False)
    return augment_datasets
