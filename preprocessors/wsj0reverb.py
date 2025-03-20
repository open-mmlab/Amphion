import json
from tqdm import tqdm
import os
import torchaudio
from utils import audio
import csv
import random
from text import _clean_text
import librosa
import soundfile as sf
import pyroomacoustics as pra
from scipy.io import wavfile
from glob import glob
from pathlib import Path
import numpy as np

SEED = 100
np.random.seed(SEED)

T60_RANGE = [0.4, 1.0]
SNR_RANGE = [0, 20]
DIM_RANGE = [5, 15, 5, 15, 2, 6]
MIN_DISTANCE_TO_WALL = 1
MIC_ARRAY_RADIUS = 0.16
TARGET_T60_SHAPE = {"CI": 0.08, "HA": 0.2}
TARGET_T60_SHAPE = {"CI": 0.10, "HA": 0.2}
TARGETS_CROP = {"CI": 16e-3, "HA": 40e-3}
NB_SAMPLES_PER_ROOM = 1
CHANNELS = 1


def obtain_clean_file(speech_list, i_sample, sample_rate=16000):
    speech, speech_sr = sf.read(speech_list[i_sample])
    speech_basename = os.path.basename(speech_list[i_sample])
    assert (
        speech_sr == sample_rate
    ), f"wrong speech sampling rate here: expected {sample_rate} got {speech_sr}"
    return speech.squeeze(), speech_sr, speech_basename[:-4]


def main(output_path, dataset_path):
    print("-" * 10)
    print("Dataset splits for {}...\n".format("wsj0reverb"))
    dataset = "wsj0reverb"
    sample_rate = 16000
    save_dir = os.path.join(output_path, dataset)
    os.makedirs(save_dir, exist_ok=True)
    wsj0reverb_path = dataset_path
    splits = ["valid", "train", "test"]
    dic_split = {"valid": "si_dt_05", "train": "si_tr_s", "test": "si_et_05"}
    speech_lists = {
        split: sorted(
            glob(f"{os.path.join(wsj0reverb_path, dic_split[split])}/**/*.wav")
        )
        for split in splits
    }

    for i_split, split in enumerate(splits):
        print("Processing split n° {}: {}...".format(i_split + 1, split))

        reverberant_output_dir = os.path.join(save_dir, "audio", split, "reverb")
        dry_output_dir = os.path.join(save_dir, "audio", split, "anechoic")
        noisy_reverberant_output_dir = os.path.join(
            save_dir, "audio", split, "noisy_reverb"
        )
        if split == "test":
            unauralized_output_dir = os.path.join(
                save_dir, "audio", split, "unauralized"
            )

        os.makedirs(reverberant_output_dir, exist_ok=True)
        os.makedirs(dry_output_dir, exist_ok=True)
        if split == "test":
            os.makedirs(unauralized_output_dir, exist_ok=True)

        speech_list = speech_lists[split]
        real_nb_samples = len(speech_list)
        for i_sample in tqdm(range(real_nb_samples)):
            if not i_sample % NB_SAMPLES_PER_ROOM:  # Generate new room
                t60 = np.random.uniform(T60_RANGE[0], T60_RANGE[1])  # Draw T60
                room_dim = np.array(
                    [
                        np.random.uniform(DIM_RANGE[2 * n], DIM_RANGE[2 * n + 1])
                        for n in range(3)
                    ]
                )  # Draw Dimensions
                center_mic_position = np.array(
                    [
                        np.random.uniform(
                            MIN_DISTANCE_TO_WALL, room_dim[n] - MIN_DISTANCE_TO_WALL
                        )
                        for n in range(3)
                    ]
                )  # draw source position
                source_position = np.array(
                    [
                        np.random.uniform(
                            MIN_DISTANCE_TO_WALL, room_dim[n] - MIN_DISTANCE_TO_WALL
                        )
                        for n in range(3)
                    ]
                )  # draw source position
                mic_array_2d = pra.beamforming.circular_2D_array(
                    center_mic_position[:-1], CHANNELS, phi0=0, radius=MIC_ARRAY_RADIUS
                )  # Compute microphone array
                mic_array = np.pad(
                    mic_array_2d,
                    ((0, 1), (0, 0)),
                    mode="constant",
                    constant_values=center_mic_position[-1],
                )

                ### Reverberant Room
                e_absorption, max_order = pra.inverse_sabine(
                    t60, room_dim
                )  # Compute absorption coeff
                reverberant_room = pra.ShoeBox(
                    room_dim,
                    fs=16000,
                    materials=pra.Material(e_absorption),
                    max_order=min(3, max_order),
                )  # Create room
                reverberant_room.set_ray_tracing()
                reverberant_room.add_microphone_array(mic_array)  # Add microphone array

            # Pick unauralized files
            speech, speech_sr, speech_basename = obtain_clean_file(
                speech_list, i_sample, sample_rate=sample_rate
            )

            # Generate reverberant room
            reverberant_room.add_source(source_position, signal=speech)
            reverberant_room.compute_rir()
            reverberant_room.simulate()
            t60_real = np.mean(reverberant_room.measure_rt60()).squeeze()
            reverberant = np.stack(reverberant_room.mic_array.signals).swapaxes(0, 1)

            e_absorption_dry = 0.99  # For Neural Networks OK but clearly not for WPE
            dry_room = pra.ShoeBox(
                room_dim,
                fs=16000,
                materials=pra.Material(e_absorption_dry),
                max_order=0,
            )  # Create room
            dry_room.add_microphone_array(mic_array)  # Add microphone array

            # Generate dry room
            dry_room.add_source(source_position, signal=speech)
            dry_room.compute_rir()
            dry_room.simulate()
            t60_real_dry = np.mean(dry_room.measure_rt60()).squeeze()
            rir_dry = dry_room.rir
            dry = np.stack(dry_room.mic_array.signals).swapaxes(0, 1)
            dry = np.pad(
                dry,
                ((0, int(0.5 * sample_rate)), (0, 0)),
                mode="constant",
                constant_values=0,
            )  # Add 1 second of silence after dry (because very dry) so that the reverb is not cut, and all samples have same length

            min_len_sample = min(reverberant.shape[0], dry.shape[0])
            dry = dry[:min_len_sample]
            reverberant = reverberant[:min_len_sample]
            output_scaling = np.max(reverberant) / 0.9

            drr = 10 * np.log10(
                np.mean(dry**2) / (np.mean(reverberant**2) + 1e-8) + 1e-8
            )
            output_filename = f"{speech_basename}_{i_sample // NB_SAMPLES_PER_ROOM}_{t60_real:.2f}_{drr:.1f}.wav"

            sf.write(
                os.path.join(dry_output_dir, output_filename),
                1 / output_scaling * dry,
                samplerate=sample_rate,
            )
            sf.write(
                os.path.join(reverberant_output_dir, output_filename),
                1 / output_scaling * reverberant,
                samplerate=sample_rate,
            )

            if split == "test":
                sf.write(
                    os.path.join(unauralized_output_dir, output_filename),
                    speech,
                    samplerate=sample_rate,
                )
