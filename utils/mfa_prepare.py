# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" This code is modified from https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/performance.html"""

import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import torchaudio
from tqdm import tqdm
from pathlib import Path


def remove_empty_dirs(path):
    """remove empty directories in a given path"""
    # Check if the given path is a directory
    if not os.path.isdir(path):
        print(f"{path} is not a directory")
        return

    # Walk through all directories and subdirectories
    for root, dirs, _ in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # Check if the directory is empty
            if not os.listdir(dir_path):
                os.rmdir(dir_path)  # "Removed empty directory


def process_single_wav_file(task):
    """process a single wav file"""
    wav_file, output_dir = task
    speaker_id, book_name, filename = Path(wav_file).parts[-3:]

    output_book_dir = Path(output_dir, speaker_id)
    output_book_dir.mkdir(parents=True, exist_ok=True)
    new_filename = f"{speaker_id}_{book_name}_{filename}"
    new_wav_file = Path(output_book_dir, new_filename)
    if not os.path.exists(new_wav_file):
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-i",
            wav_file,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            new_wav_file,
        ]
        subprocess.check_call(
            command
        )  # Run the command to convert the file to 16kHz and 16-bit PCM
        os.remove(wav_file)
    else:
        print(f"Skipping {new_wav_file} as it already exists")


def process_wav_files(wav_files, output_dir, n_process):
    """process wav files in parallel"""
    tasks = [(wav_file, output_dir) for wav_file in wav_files]
    print(f"Processing {len(tasks)} files")
    with Pool(processes=n_process) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_single_wav_file, tasks), total=len(tasks)
        ):
            pass
    print("Removing empty directories...")
    remove_empty_dirs(output_dir)
    print("Done!")


def get_wav_files(directory):
    """get all wav files in the dataset"""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    print("total wav files: {}".format(len(wav_files)))
    return wav_files


def filter_wav_files_by_length(wav_files, max_len_sec=15):
    """filter wav files by length"""
    print("original wav files: {}".format(len(wav_files)))
    filtered_wav_files = []
    print("filtering wav files by length...")
    for audio_file in tqdm(wav_files):
        metadata = torchaudio.info(str(audio_file))
        audio_length = metadata.num_frames / metadata.sample_rate
        if audio_length <= max_len_sec:
            filtered_wav_files.append(audio_file)
        else:
            os.remove(audio_file)
    print("filtered wav files: {}".format(len(filtered_wav_files)))
    return filtered_wav_files


if __name__ == "__main__":
    dataset_path = "/path/to/output/directory"
    n_process = 16
    max_len_sec = 15
    wav_files = get_wav_files(dataset_path)
    filtered_wav_files = filter_wav_files_by_length(wav_files, max_len_sec)
    process_wav_files(filtered_wav_files, dataset_path, n_process)
