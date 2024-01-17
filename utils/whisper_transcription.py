# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import string
import time
from multiprocessing import Pool, Value, Lock
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import whisper

processed_files_count = Value("i", 0)  # count of processed files
lock = Lock()  # lock for the count


def preprocess_text(text):
    """Preprocess text after ASR"""
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def transcribe_audio(model, processor, audio_file, device):
    """Transcribe audio file"""
    audio = whisper.load_audio(audio_file)  # load from path
    audio = whisper.pad_or_trim(audio)  # default 30 seconds
    inputs = whisper.log_mel_spectrogram(audio).to(
        device=device
    )  # convert to spectrogram
    inputs = inputs.unsqueeze(0).type(torch.cuda.HalfTensor)  # add batch dimension

    outputs = model.generate(
        inputs=inputs, max_new_tokens=128
    )  # generate transcription
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[
        0
    ]  # decode
    transcription_processed = preprocess_text(transcription)  # preprocess
    return transcription_processed


def write_transcription(audio_file, transcription):
    """Write transcription to txt file"""
    txt_file = audio_file.with_suffix(".txt")
    with open(txt_file, "w") as file:
        file.write(transcription)


def init_whisper(model_id, device):
    """Initialize whisper model and processor"""
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model {model_id}")  # model_id = "distil-whisper/distil-large-v2"
    distil_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=False
    )
    distil_model = distil_model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return distil_model, processor


def asr_wav_files(file_list, gpu_id, total_files, model_id):
    """Transcribe wav files in a list"""
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    whisper_model, processor = init_whisper(model_id, device)
    print(f"Processing on {device} starts")
    start_time = time.time()
    for audio_file in file_list:
        try:
            transcription = transcribe_audio(
                whisper_model, processor, audio_file, device
            )
            write_transcription(audio_file, transcription)
            with lock:
                processed_files_count.value += 1
                if processed_files_count.value % 5 == 0:
                    current_time = time.time()
                    avg_time_per_file = (current_time - start_time) / (
                        processed_files_count.value
                    )
                    remaining_files = total_files - processed_files_count.value
                    estimated_time_remaining = avg_time_per_file * remaining_files
                    remaining_time_formatted = time.strftime(
                        "%H:%M:%S", time.gmtime(estimated_time_remaining)
                    )
                    print(
                        f"Processed {processed_files_count.value}/{total_files} files, time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, Estimated time remaining: {remaining_time_formatted}"
                    )
        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")


def asr_main(input_dir, num_gpus, model_id):
    """Transcribe wav files in a directory"""
    num_processes = min(num_gpus, os.cpu_count())
    print(f"Using {num_processes} GPUs for transcription")
    wav_files = list(pathlib.Path(input_dir).rglob("*.wav"))
    total_files = len(wav_files)
    print(f"Found {total_files} wav files in {input_dir}")
    files_per_process = len(wav_files) // num_processes
    print(f"Processing {files_per_process} files per process")
    with Pool(num_processes) as p:
        p.starmap(
            asr_wav_files,
            [
                (
                    wav_files[i * files_per_process : (i + 1) * files_per_process],
                    i % num_gpus,
                    total_files,
                    model_id,
                )
                for i in range(num_processes)
            ],
        )
    print("Done!")


if __name__ == "__main__":
    input_dir = "/path/to/output/directory"
    num_gpus = 2
    model_id = "distil-whisper/distil-large-v2"
    asr_main(input_dir, num_gpus, model_id)
