# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import os
import torchaudio
import torch
import textgrid
import numpy as np

from utils.mfa_prepare import (
    process_wav_files,
    get_wav_files,
    filter_wav_files_by_length,
)
from utils.cut_by_vad import cut_segments
from utils.whisper_transcription import asr_main, get_txt_files
from utils.util import has_existed

import subprocess
import random
from collections import defaultdict
from glob import glob


def librilight_statistics(data_dir, speaker2split):
    """Get statistics for librilight dataset"""
    distribution2speakers2utts = defaultdict(lambda: defaultdict(list))
    speaker_infos = glob(data_dir + "/*")
    if len(speaker_infos) == 0:
        raise Exception("No speaker info found.")
    for speaker_info in speaker_infos:
        speaker = speaker_info.split("/")[-1]
        distribution = speaker2split[speaker]
        utts = glob(speaker_info + "/*.wav")
        for utt in utts:
            uid = utt.split("/")[-1].split(".")[0]
            distribution2speakers2utts[distribution][speaker].append(uid)
    return distribution2speakers2utts


def get_speakers_from_directory(directory):
    return [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]


def get_duration_phone_start_end(textgrid_path):
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    phone_tier = tg[0]
    durations = []
    phones = []
    starts = []
    ends = []
    for interval in phone_tier:
        start = interval.minTime
        end = interval.maxTime
        phone = interval.mark
        duration = end - start
        phones.append(phone)
        durations.append(duration)
        starts.append(start)
        ends.append(end)
    return durations, phones, starts, ends


def duration2feature(durations, sr=16000, hop_size=200):
    # durations to features: seconds*sample rate/hopsize
    durations_features = []
    for duration in durations:
        durations_features.append(int(duration * sr / hop_size))
    return durations_features


def split_dataset_by_speaker(base_dir, train_ratio=0.6, dev_ratio=0.2):
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

    unique_speakers = list(set(all_speakers))
    unique_speakers.sort()

    # Get speaker2split dictionary
    speaker2split = {}
    for speaker in unique_speakers:
        if speaker in train_speakers:
            speaker2split[speaker] = "train"
        elif speaker in dev_speakers:
            speaker2split[speaker] = "dev"
        elif speaker in eval_speakers:
            speaker2split[speaker] = "eval"
        else:
            raise Exception("Speaker not found in any split.")

    return unique_speakers, speaker2split


def save_meta_data(save_dir, processed_dir, distribution2speakers2utts, speakers):
    """Save metadata for librilight dataset"""
    os.makedirs(save_dir, exist_ok=True)
    print("Saving metadata to", save_dir)
    train_output_file = os.path.join(save_dir, "train.json")
    valid_output_file = os.path.join(save_dir, "dev.json")
    test_output_file = os.path.join(save_dir, "eval.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    duration_dir = os.path.join(save_dir, "Durations")
    os.makedirs(duration_dir, exist_ok=True)
    phone_dir = os.path.join(save_dir, "Phones")
    os.makedirs(phone_dir, exist_ok=True)
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    utt2singer = open(utt2singer_file, "w")
    if has_existed(train_output_file):
        # show save dir
        print("Metadata already exists in", save_dir)
        print("Skipping...")
        return

    train = []
    test = []
    valid = []

    train_index_count = 0
    test_index_count = 0
    valid_index_count = 0

    train_total_length = 0
    test_total_length = 0
    valid_total_length = 0

    # Save metadata
    for distribution, speakers2utts in tqdm(distribution2speakers2utts.items()):
        for speaker, utts in tqdm(speakers2utts.items()):
            for chosen_uid in utts:
                res = {
                    "Dataset": "librilight",
                    "Singer": speaker,
                    "Uid": "{}#{}#{}".format(distribution, speaker, chosen_uid),
                    "Distribution": distribution,
                }
                res["Path"] = "{}/{}.wav".format(speaker, chosen_uid)
                res["Path"] = os.path.join(processed_dir, res["Path"])
                text_file_path = os.path.join(
                    processed_dir,
                    speaker,
                    chosen_uid + ".txt",
                )
                textgrid_file_path = os.path.join(
                    processed_dir,
                    speaker,
                    chosen_uid + ".TextGrid",
                )

                # if text_path textgrid_path and wav_path not all exist, skip
                if (
                    not os.path.exists(text_file_path)
                    or not os.path.exists(textgrid_file_path)
                    or not os.path.exists(res["Path"])
                ):
                    continue

                with open(text_file_path, "r") as f:
                    lines = f.readlines()
                    assert len(lines) == 1
                    text = lines[0].strip()
                    res["Text"] = text

                durations, phones, starts, ends = get_duration_phone_start_end(
                    textgrid_file_path
                )
                durations_features = duration2feature(durations)
                # save durations as .npy to duration dir
                np.save(
                    os.path.join(duration_dir, res["Uid"] + ".npy"), durations_features
                )

                # save phones as .npy to phone dir
                np.save(os.path.join(phone_dir, res["Uid"] + ".npy"), phones)

                res["Start"] = starts[0]
                res["End"] = ends[-1]

                metainfo = torchaudio.info(res["Path"])
                length = metainfo.num_frames / metainfo.sample_rate
                res["Length"] = length

                if "train" in distribution:
                    res["index"] = train_index_count
                    train_total_length += length
                    train.append(res)
                    train_index_count += 1
                elif "dev" in distribution:
                    res["index"] = valid_index_count
                    valid_total_length += length
                    valid.append(res)
                    valid_index_count += 1
                elif "eval" in distribution:
                    res["index"] = test_index_count
                    test_total_length += length
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
        "#Train length= {}, #Dev length= {}, #Eval length= {}".format(
            train_total_length / 3600,
            valid_total_length / 3600,
            test_total_length / 3600,
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
    mfa_dict_path = cfg.mfa_dict_path  # path to mfa dict file
    mfa_model_path = cfg.mfa_model_path  # path to mfa model file
    mfa_config_path = cfg.mfa_config_path  # path to mfa config file

    # check if mfa files exist
    if not os.path.exists(mfa_dict_path) or not os.path.exists(mfa_model_path):
        raise Exception("MFA files not found.")

    have_mfa_config = False
    if len(mfa_config_path) == 0:
        pass
    else:
        if os.path.exists(mfa_config_path):
            have_mfa_config = True

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
    used_subsets = cfg.used_subsets
    print("Found subsets:", subsets)
    subsets = [s for s in subsets if s in used_subsets]
    print("Using subsets:", used_subsets)

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

        processed_wav = get_wav_files(processed_dir)
        processed_wav_num = len(processed_wav)

        if processed_wav_num == 0:
            cut_segments(raw_dir, processed_dir, cut_length, n_cpus)
        else:
            print("Audio files already cut. Skipping...")

        # Steps 2 & 3: Filter and Preprocess
        print("-" * 10)
        print("Step 2 & 3: Filter and Preprocess")
        print("Filtering and preprocessing audio files...")

        wav_files = get_wav_files(processed_dir)
        filtered_wav_files = filter_wav_files_by_length(wav_files, max_length)
        # if number of wav files in processed_dir == filtered_wav_files, filtering is not needed
        if len(wav_files) != len(filtered_wav_files):
            process_wav_files(filtered_wav_files, processed_dir, n_cpus)
        else:
            print("Audio files already filtered. Skipping...")

        # Step 4 & 5: Transcription & Text-preprocess
        print("-" * 10)
        print("Step 4 & 5: Transcription & Text-preprocess")
        print("Transcribing audio files...")

        n_gpus = min(n_gpus, torch.cuda.device_count())
        # number of transcripted txt
        txt_files = get_txt_files(processed_dir)
        # if number of wav files in processed_dir == number of txt files in processed_dir, transcription is not needed
        if len(wav_files) != len(txt_files):
            asr_main(processed_dir, n_gpus, model_id)
        else:
            print("Audio files already transcribed. Skipping...")

        # Step 6: MFA Align
        print("-" * 10)
        print("Step 6: MFA Align")
        print("Aligning audio files...")

        if have_mfa_config == True:
            align_command = f"""mfa align -c {mfa_config_path} --output_format long_textgrid --clean --overwrite -v -j {str(n_cpus)} {processed_dir} {mfa_dict_path} {mfa_model_path} {processed_dir}"""
        else:
            align_command = f"""mfa align --output_format long_textgrid --clean --overwrite -v -j {str(n_cpus)} {processed_dir} {mfa_dict_path} {mfa_model_path} {processed_dir}"""
        process = subprocess.Popen(
            align_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
        _ = process.poll()

        print("Aligning done!")

        # Step 7: train/dev/eval split
        print("-" * 10)
        print("Step 7: train/dev/eval split")
        print("Splitting dataset by speaker...")

        speakers, speaker2split = split_dataset_by_speaker(processed_dir)

        # Step 8: Statistics
        print("-" * 10)
        print("Step 8: Statistics")
        print("Calculating statistics...")

        distribution2speakers2utts = librilight_statistics(processed_dir, speaker2split)

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
