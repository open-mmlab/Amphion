import json
import os
import io
import scipy
import os
import shutil
from pydub import AudioSegment
import soundfile as sf


def load_cfg(cfg_path):
    if not os.path.exists("config.json"):
        raise FileNotFoundError(
            "config.json not found. Please: copy, config, and rename `config.json.example` to `config.json`"
        )
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return cfg


def write_wav(path, sr, x):
    """numpy array to WAV"""
    sf.write(path, x, sr)


def write_mp3(path, sr, x):
    """numpy array to MP3"""
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, sr, x)
    wav_io.seek(0)
    sound = AudioSegment.from_wav(wav_io)
    with open(path, "wb") as af:
        sound.export(
            af,
            format="mp3",
            codec="mp3",
            bitrate="160000",
        )


# 读取文件夹内所有音频文件
def get_audio_files(folder_path):
    audio_files = []
    for root, _, files in os.walk(folder_path):
        if "_processed" in root:
            continue
        for file in files:
            if ".temp" in file:
                continue
            if file.endswith((".mp3", ".wav", ".flac", ".m4a")):
                audio_files.append(os.path.join(root, file))
    return audio_files


def get_specific_files(folder_path, ext):
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


def move_vocals(src_directory):
    # 遍历根目录下的所有文件和文件夹
    for root, _, files in os.walk(src_directory):
        for file in files:
            # 检查文件名是否为'vocals.mp3'
            if file == "vocals.mp3":
                # 构建源文件的完整路径
                src_path = os.path.join(root, file)
                # 获取父级目录的名称
                parent_dir_name = os.path.basename(root)
                # 构建目标文件的完整路径
                dest_path = os.path.join(src_directory, parent_dir_name + ".mp3")
                # 复制文件
                shutil.copy(src_path, dest_path)

    # 删除源文件夹
    shutil.rmtree(src_directory + "/htdemucs")
