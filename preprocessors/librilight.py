# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import os
import torchaudio
import torch


from utils.mfa_prepare import (
    process_wav_files,
    get_wav_files,
    filter_wav_files_by_length,
)
from utils.cut_by_vad import cut_segments
from utils.whisper_transcription import asr_main
from utils.util import has_existed

import subprocess
import random
from collections import defaultdict
from glob import glob
import shutil


def librilight_statistics(data_dir):
    """Get statistics for librilight dataset"""
    distribution2speakers2utts = defaultdict(lambda: defaultdict(list))
    distribution_infos = glob(data_dir + "/*")
    for distribution_info in distribution_infos:
        distribution = distribution_info.split("/")[-1]
        print(distribution)
        speaker_infos = glob(distribution_info + "/*")
        if len(speaker_infos) == 0:
            continue
        for speaker_info in speaker_infos:
            speaker = speaker_info.split("/")[-1]
            utts = glob(speaker_info + "/*.wav")
            for utt in utts:
                uid = utt.split("/")[-1].split(".")[0]
                distribution2speakers2utts[distribution][speaker].append(uid)
    return distribution2speakers2utts


def get_speakers_from_directory(directory):
    return [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]


def split_dataset_by_speaker(base_dir, train_ratio=0.8, dev_ratio=0.1):
    train_dir = os.path.join(base_dir, "train")
    dev_dir = os.path.join(base_dir, "dev")
    eval_dir = os.path.join(base_dir, "eval")

    # Check if dataset is already split
    if has_existed(train_dir) or has_existed(dev_dir) or has_existed(eval_dir):
        print("Dataset already split. Calculating speakers...")
        train_speakers = get_speakers_from_directory(train_dir)
        dev_speakers = get_speakers_from_directory(dev_dir)
        eval_speakers = get_speakers_from_directory(eval_dir)
        all_speakers = train_speakers + dev_speakers + eval_speakers
        unique_speakers = list(set(all_speakers))
        unique_speakers.sort()
        return unique_speakers

    # List all directories in the base directory
    all_speakers = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    random.shuffle(all_speakers)

    # Calculate split sizes
    total_speakers = len(all_speakers)
    train_size = int(total_speakers * train_ratio)
    dev_size = int(total_speakers * dev_ratio)
    eval_size = total_speakers - train_size - dev_size
    print("Total speakers:", total_speakers)
    print("Train speakers:", train_size)
    print("Dev speakers:", dev_size)
    print("Eval speakers:", eval_size)

    # Split directories
    train_speakers = all_speakers[:train_size]
    dev_speakers = all_speakers[train_size : train_size + dev_size]
    eval_speakers = all_speakers[train_size + dev_size :]

    # Function to move directories
    def move_speakers(speakers, target_dir):
        for speaker in speakers:
            shutil.move(
                os.path.join(base_dir, speaker), os.path.join(target_dir, speaker)
            )

    # Move directories
    print("Moving directories...")
    print("Moving Train speakers...")
    move_speakers(train_speakers, train_dir)
    print("Moving Dev speakers...")
    move_speakers(dev_speakers, dev_dir)
    print("Moving Eval speakers...")
    move_speakers(eval_speakers, eval_dir)

    unique_speakers = list(set(all_speakers))
    unique_speakers.sort()
    return unique_speakers


def save_meta_data(save_dir, processed_dir, distribution2speakers2utts, speakers):
    """Save metadata for librilight dataset"""
    os.makedirs(save_dir, exist_ok=True)
    train_output_file = os.path.join(save_dir, "train.json")
    valid_output_file = os.path.join(save_dir, "dev.json")
    test_output_file = os.path.join(save_dir, "eval.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    utt2singer = open(utt2singer_file, "w")
    if has_existed(train_output_file):
        print("Metadata already exists. Skipping...")
        return

    train = []
    test = []
    valid = []

    train_index_count = 0
    test_index_count = 0
    valid_index_count = 0

    train_total_duration = 0
    test_total_duration = 0
    valid_total_duration = 0

    # Save metadata
    for distribution, speakers2utts in tqdm(distribution2speakers2utts.items()):
        for speaker, utts in tqdm(speakers2utts.items()):
            for chosen_uid in utts:
                res = {
                    "Dataset": "librilight",
                    "Singer": speaker,
                    "Uid": "{}#{}#{}".format(distribution, speaker, chosen_uid),
                }
                res["Path"] = "{}/{}/{}.wav".format(distribution, speaker, chosen_uid)
                res["Path"] = os.path.join(processed_dir, res["Path"])
                assert os.path.exists(res["Path"])

                text_file_path = os.path.join(
                    processed_dir,
                    distribution,
                    speaker,
                    chosen_uid + ".txt",
                )
                with open(text_file_path, "r") as f:
                    lines = f.readlines()
                    assert len(lines) == 1
                    text = lines[0].strip()
                    res["Text"] = text

                waveform, sample_rate = torchaudio.load(res["Path"])
                duration = waveform.size(-1) / sample_rate
                res["Duration"] = duration

                if "train" in distribution:
                    res["index"] = train_index_count
                    train_total_duration += duration
                    train.append(res)
                    train_index_count += 1
                elif "dev" in distribution:
                    res["index"] = valid_index_count
                    valid_total_duration += duration
                    valid.append(res)
                    valid_index_count += 1
                elif "eval" in distribution:
                    res["index"] = test_index_count
                    test_total_duration += duration
                    test.append(res)
                    test_index_count += 1
                utt2singer.write("{}\t{}\n".format(res["Uid"], res["Singer"]))
    print("Done!")
    print(
        "Utterance count: train = {}, dev = {}, eval = {}".format(
            len(train), len(valid), len(test)
        )
    )
    print(
        "#Train duration= {}, #Dev duration= {}, #Eval duration= {}".format(
            train_total_duration / 3600,
            valid_total_duration / 3600,
            test_total_duration / 3600,
        )
    )
    with open(train_output_file, "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(test_output_file, "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
    with open(valid_output_file, "w") as f:
        json.dump(valid, f, indent=4, ensure_ascii=False)
    utt2singer.close()
    singer_lut = {name: i for i, name in enumerate(speakers)}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)
    print("Metadata saved to", save_dir)


def main(output_path, dataset_path, cfg):
    """Preprocess librilight dataset"""
    n_cpus = cfg.n_cpus  # number of cpus to use for preprocessing
    n_gpus = cfg.n_gpus  # number of gpus to use for transcription
    cut_length = cfg.cut_length  # target length of utterance in seconds
    max_length = cfg.max_length  # max length of utterance in seconds

    # MFA files
    mfa_config_path = cfg.mfa_config_path  # path to mfa config file
    mfa_dict_path = cfg.mfa_dict_path  # path to mfa dict file
    mfa_model_path = cfg.mfa_model_path  # path to mfa model file

    # check if mfa files exist
    if (
        not os.path.exists(mfa_dict_path)
        or not os.path.exists(mfa_model_path)
        or not os.path.exists(mfa_config_path)
    ):
        raise Exception("MFA files not found.")

    # Whisper model id
    model_id = cfg.whisper_model_id  # id of whisper model to use for transcription

    subsets = [
        d
        for d in os.listdir(dataset_path)
        if (
            os.path.isdir(os.path.join(dataset_path, d))
            and d in ["tiny", "small", "medium", "large"]
        )
    ]
    print("Found subsets:", subsets)

    if len(subsets) == 0:
        print("No subsets found. Exiting...")
        return
    # Preprocess each subset
    for subset in subsets:
        # Construct paths based on the base path
        print("Pre-proccessing Libri-light subset:", subset)
        raw_dir = f"{dataset_path}/{subset}"
        save_dir = f"{output_path}/{subset}"
        processed_dir = f"{dataset_path}/processed/{subset}"
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # Step 1: Segmentation
        print("-" * 10)
        print("Step 1: Segmentation")
        print("Cutting audio files...")

        cut_segments(raw_dir, processed_dir, cut_length, n_cpus)

        # Steps 2 & 3: Filter and Preprocess
        print("-" * 10)
        print("Step 2 & 3: Filter and Preprocess")
        print("Filtering and preprocessing audio files...")

        wav_files = get_wav_files(processed_dir)
        filtered_wav_files = filter_wav_files_by_length(wav_files, max_length)
        process_wav_files(filtered_wav_files, processed_dir, n_cpus)

        # Step 4 & 5: Transcription & Text-preprocess
        print("-" * 10)
        print("Step 4 & 5: Transcription & Text-preprocess")
        print("Transcribing audio files...")

        n_gpus = min(n_gpus, torch.cuda.device_count())
        asr_main(processed_dir, n_gpus, model_id)

        # Step 6: MFA Align
        print("-" * 10)
        print("Step 6: MFA Align")
        print("Aligning audio files...")

        command = [
            "mfa",
            "align",
            "-v",
            "-j",
            str(n_cpus),
            "-c",
            mfa_config_path,
            processed_dir,
            mfa_dict_path,
            mfa_model_path,
            processed_dir,
            "--output_format",
            "long_textgrid",
            "--clean",
            "--overwrite",
        ]
        subprocess.run(command, text=True)

        # Step 7: train/dev/eval split
        print("-" * 10)
        print("Step 7: train/dev/eval split")
        print("Splitting dataset by speaker...")

        speakers = split_dataset_by_speaker(processed_dir)

        # Step 8: Statistics
        print("-" * 10)
        print("Step 8: Statistics")
        print("Calculating statistics...")

        distribution2speakers2utts = librilight_statistics(processed_dir)

        # Step 9: Save metadata
        print("-" * 10)
        print("Step 9: Save metadata")
        print("Preparing Metadata for Librilight...")

        save_meta_data(save_dir, processed_dir, distribution2speakers2utts, speakers)
        print("Preprocessing subset", subset, "done!")
        print("-" * 10)


if __name__ == "__main__":
    dataset_path = "/path/to/dataset/librilight"
    output_path = "/path/to/output"
    main(output_path, dataset_path)
