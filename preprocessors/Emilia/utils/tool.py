# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor
import json
import os
import librosa
import numpy as np
import time
import torch
from pydub import AudioSegment
import soundfile as sf
import onnxruntime as ort
import tqdm
import subprocess
import re

from utils.logger import Logger, time_logger


def load_cfg(cfg_path):
    """
    Load configuration from a JSON file.

    Args:
        cfg_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found. Please copy, configure, and rename `config.json.example` to `{cfg_path}`."
        )
    with open(cfg_path, "r") as f:
        try:
            cfg = json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise TypeError(
                "Please finish the `// TODO:` in the `config.json` file before running the script. Check README.md for details."
            )
    return cfg


def write_wav(path, sr, x):
    """Write numpy array to WAV file."""
    sf.write(path, x, sr)


def write_mp3(path, sr, x):
    """Convert numpy array to MP3."""
    try:
        # Ensure x is in the correct format and normalize if necessary
        if x.dtype != np.int16:
            # Normalize the array to fit in int16 range if it's not already int16
            x = np.int16(x / np.max(np.abs(x)) * 32767)

        # Create audio segment from numpy array
        audio = AudioSegment(
            x.tobytes(), frame_rate=sr, sample_width=x.dtype.itemsize, channels=1
        )
        # Export as MP3 file
        audio.export(path, format="mp3")
    except Exception as e:
        print(e)
        print("Error: Failed to write MP3 file.")


def get_audio_files(folder_path):
    """Get all audio files in a folder."""
    audio_files = []
    for root, _, files in os.walk(folder_path):
        if "_processed" in root:
            continue
        for file in files:
            if ".temp" in file:
                continue
            if file.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
                audio_files.append(os.path.join(root, file))
    return audio_files


def get_specific_files(folder_path, ext):
    """Get specific files with a given extension in a folder."""
    audio_files = []
    for root, _, files in os.walk(folder_path):
        if "_processed" in root:
            continue
        for file in files:
            if ".temp" in file:
                continue
            if file.endswith(ext):
                audio_files.append(os.path.join(root, file))
    return audio_files


def export_to_srt(asr_result, file_path):
    """Export ASR result to SRT file."""
    with open(file_path, "w") as f:

        def format_time(seconds):
            return (
                time.strftime("%H:%M:%S", time.gmtime(seconds))
                + f",{int(seconds * 1000 % 1000):03d}"
            )

        for idx, segment in enumerate(asr_result):
            f.write(f"{idx + 1}\n")
            f.write(
                f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n"
            )
            f.write(f"{segment['speaker']}: {segment['text']}\n\n")


def detect_gpu():
    """Detect if GPU is available and print related information."""
    logger = Logger.get_logger()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logger.info("ENV: CUDA_VISIBLE_DEVICES not set, use default setting")
    else:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info(f"ENV: CUDA_VISIBLE_DEVICES = {gpu_id}")

    if not torch.cuda.is_available():
        logger.error("Torch CUDA: No GPU detected. torch.cuda.is_available() = False.")
        return False

    num_gpus = torch.cuda.device_count()
    logger.debug(f"Torch CUDA: Detected {num_gpus} GPUs.")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        logger.debug(f" * GPU {i}: {gpu_name}")

    logger.debug("Torch: CUDNN version = " + str(torch.backends.cudnn.version()))
    if not torch.backends.cudnn.is_available():
        logger.error("Torch: CUDNN is not available.")
        return False
    logger.debug("Torch: CUDNN is available.")

    ort_providers = ort.get_available_providers()
    logger.debug(f"ORT: Available providers: {ort_providers}")
    if "CUDAExecutionProvider" not in ort_providers:
        logger.warning(
            "ORT: CUDAExecutionProvider is not available. "
            "Please install a compatible version of ONNX Runtime. "
            "See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html"
        )

    return True


def get_gpu_nums():
    """Get GPU nums by nvidia-smi."""
    logger = Logger.get_logger()
    try:
        result = subprocess.check_output("nvidia-smi -L | wc -l", shell=True)
        gpus_count = int(result.decode().strip())
    except Exception as e:
        logger.error("Error occurred while getting GPU count: " + str(e))
        gpus_count = 8  # Default to 8 if GPU count retrieval fails
    return gpus_count


def check_env(logger):
    """Check environment variables."""
    if "http_proxy" in os.environ:
        logger.info(f"ENV: http_proxy = {os.environ['http_proxy']}")
    else:
        logger.info("ENV: http_proxy not set")

    if "https_proxy" in os.environ:
        logger.info(f"ENV: https_proxy = {os.environ['https_proxy']}")
    else:
        logger.info("ENV: https_proxy not set")

    if "HF_ENDPOINT" in os.environ:
        logger.info(
            f"ENV: HF_ENDPOINT = {os.environ['HF_ENDPOINT']}, if downloading slow, try `unset HF_ENDPOINT`"
        )
    else:
        logger.info("ENV: HF_ENDPOINT not set")

    hostname = os.popen("hostname").read().strip()
    logger.debug(f"HOSTNAME: {hostname}")

    environ_path = os.environ["PATH"]
    environ_ld_library = os.environ.get("LD_LIBRARY_PATH", "")
    logger.debug(f"ENV: PATH = {environ_path}, LD_LIBRARY_PATH = {environ_ld_library}")


@time_logger
def export_to_mp3(audio, asr_result, folder_path, file_name):
    """Export segmented audio to MP3 files."""
    sr = audio["sample_rate"]
    audio = audio["waveform"]

    os.makedirs(folder_path, exist_ok=True)

    # Function to process each segment in a separate thread
    def process_segment(idx, segment):
        start, end = int(segment["start"] * sr), int(segment["end"] * sr)
        split_audio = audio[start:end]
        split_audio = librosa.to_mono(split_audio)
        out_file = f"{file_name}_{idx}.mp3"
        out_path = os.path.join(folder_path, out_file)
        write_mp3(out_path, sr, split_audio)

    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=72) as executor:
        # Submit each segment processing as a separate thread
        futures = [
            executor.submit(process_segment, idx, segment)
            for idx, segment in enumerate(asr_result)
        ]

        # Wait for all threads to complete
        for future in tqdm.tqdm(
            futures, total=len(asr_result), desc="Exporting to MP3"
        ):
            future.result()


@time_logger
def export_to_wav(audio, asr_result, folder_path, file_name):
    """Export segmented audio to WAV files."""
    sr = audio["sample_rate"]
    audio = audio["waveform"]

    os.makedirs(folder_path, exist_ok=True)

    for idx, segment in enumerate(tqdm.tqdm(asr_result, desc="Exporting to WAV")):
        start, end = int(segment["start"] * sr), int(segment["end"] * sr)
        split_audio = audio[start:end]
        split_audio = librosa.to_mono(split_audio)
        out_file = f"{file_name}_{idx}.wav"
        out_path = os.path.join(folder_path, out_file)
        write_wav(out_path, sr, split_audio)


def get_char_count(text):
    """
    Get the number of characters in the text.

    Args:
        text (str): Input text.

    Returns:
        int: Number of characters in the text.
    """
    # Using regular expression to remove punctuation and spaces
    cleaned_text = re.sub(r"[,.!?\"'，。！？“”‘’ ]", "", text)
    char_count = len(cleaned_text)
    return char_count


def calculate_audio_stats(
    data, min_duration=3, max_duration=30, min_dnsmos=3, min_char_count=2
):
    """
    Reading the proviced json, calculate and return the audio ID and their duration that meet the given filtering criteria.

    Args:
        data: JSON.
        min_duration: Minimum duration of the audio in seconds.
        max_duration: Maximum duration of the audio in seconds.
        min_dnsmos: Minimum DNSMOS value.
        min_char_count: Minimum number of characters.

    Returns:
        valid_audio_stats: A list containing tuples of audio ID and their duration.
    """
    all_audio_stats = []
    valid_audio_stats = []
    avg_durations = []

    # iterate over each entry in the JSON to collect the average duration of the phonemes
    for entry in data:
        # remove punctuation and spaces
        char_count = get_char_count(entry["text"])
        duration = entry["end"] - entry["start"]
        if char_count > 0:
            avg_durations.append(duration / char_count)

    # calculate the bounds for the average character duration
    if len(avg_durations) > 0:
        q1 = np.percentile(avg_durations, 25)
        q3 = np.percentile(avg_durations, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    else:
        # if no valid character data, use default values
        lower_bound, upper_bound = 0, np.inf

    # iterate over each entry in the JSON to apply all filtering criteria
    for idx, entry in enumerate(data):
        duration = entry["end"] - entry["start"]
        dnsmos = entry["dnsmos"]
        # remove punctuation and spaces
        char_count = get_char_count(entry["text"])
        if char_count > 0:
            avg_char_duration = duration / char_count
        else:
            avg_char_duration = 0

        # collect the duration of all audios
        all_audio_stats.append((idx, duration))

        # apply filtering criteria
        if (
            (min_duration <= duration <= max_duration)  # withing duration range
            and (dnsmos >= min_dnsmos)
            and (char_count >= min_char_count)
            and (
                lower_bound <= avg_char_duration <= upper_bound
            )  # average character duration within bounds
        ):
            valid_audio_stats.append((idx, duration))

    return valid_audio_stats, all_audio_stats
