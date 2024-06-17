# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from utils.data_utils import *
from tqdm import tqdm
import librosa
from petrel_client.client import Client
from torch.utils.data import Dataset
import pandas as pd
import rir_generator as rir
import time
import io
from multiprocessing import Pool, Lock

NUM_WORKERS = 32
lock = Lock()
SAMPLE_RATE = 16000


def get_duration(file_path):
    duration = librosa.get_duration(path=file_path, sr=SAMPLE_RATE)
    return file_path, duration


# g2p
# from utils.g2p.g2p import phonemizer_g2p

# override g2p with g2p_en library
from .g2p_processor import G2pProcessor

phonemizer_g2p = G2pProcessor()

# lang2token ={
#     'zh': "[ZH]",
#     'ja':"[JA]",
#     "en":"[EN]",
#     "fr":"[FR]",
#     "kr": "[KR]",
#     "de": "[DE]",
# }
# LANG2CODE = {
#     'en': 655,
#     'zh': 654,
# }
import logging


class PhonemizerWarningFilter(logging.Filter):
    def filter(self, record):
        # 只过滤 phonemizer 中的 WARNING 级别日志
        if record.name == "phonemizer" and record.levelno == logging.WARNING:
            return False
        return False


logger = logging.getLogger("phonemizer")
filter = PhonemizerWarningFilter()
logger.addFilter(filter)
logging.basicConfig(level=logging.INFO)


class VALLEDataset(Dataset):
    def __init__(self, args, is_valid=False, resample_to_24k=False):
        print(f"Initializing VALLEDataset")
        dataset_list = args.dataset_list
        dataset_cache_dir = args.cache_dir  # cache_dir
        print(f"args.cache_dir = ", args.cache_dir)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        # create dataset2dir

        self.client = Client("/mnt/petrelfs/hehaorui/petreloss.conf")
        self.resample_to_24k = resample_to_24k
        if self.resample_to_24k:
            assert SAMPLE_RATE == 24000
            print(f"Using 24k resampling.")

        print(f"data sampling rate is {SAMPLE_RATE}")

        self.dataset2dir = {
            "mls_train": "public-dataset-p2:s3://public-dataset-p2/Multilingual-LibriSpeech/data_0321/unzip/mls_english1/train/audio",
            "mls_dev": "public-dataset-p2:s3://public-dataset-p2/Multilingual-LibriSpeech/data_0321/unzip/mls_english1/dev/audio",
            "mls_test": "public-dataset-p2:s3://public-dataset-p2/Multilingual-LibriSpeech/data_0321/unzip/mls_english1/test/audio",
            "librilight_small": "amphion:s3://amphion/Libri-light/small_15s",
            "librilight_medium": "amphion:s3://amphion/Libri-light/medium_15s",
            "librilight_large": "amphion:s3://amphion/Libri-light/large_15s",
            "mls_german": "public-dataset-p2:s3://public-dataset-p2/Multilingual-LibriSpeech/data_0321/unzip/mls_german/train/audio",
        }

        self.use_speaker = args.use_speaker
        self.use_noise = args.use_noise
        print(f"Using speaker: {self.use_speaker}, using noise: {self.use_noise}")

        self.dataset_list = dataset_list
        self.meta_data_cache = None

        self.transcripts = None

        for dataset_name in self.dataset_list:
            if dataset_name == "mls_train":
                self.meta_data_cache_path = os.path.join(
                    dataset_cache_dir, "mls_train_metadata_cache.csv"
                )
                # read meta data cache: MAIN_metadata_cache.csv
                print(f"Loaded metadata cache from {self.meta_data_cache_path}")

                # write language info
                tmp_cache = pd.read_csv(self.meta_data_cache_path, encoding="utf-8")
                tmp_cache["language"] = "en"

                if self.meta_data_cache == None:
                    self.meta_data_cache = tmp_cache
                else:
                    self.meta_data_cache.append(tmp_cache)

                if len(self.meta_data_cache) == 0:
                    print(f"Empty metadata cache!")
                    raise ValueError("Empty metadata cache!")
                elif len(self.meta_data_cache) < 10731070:
                    print(f"Need to reload metadata cache!")
                    print(f"Current size: {len(self.meta_data_cache)}")
                    raise ValueError("Need to reload metadata cache!")
                print(f"Loaded {len(self.meta_data_cache)} metadata_cache")

                import pickle

                # load mls en transcripts
                if not os.path.isfile(
                    "/mnt/petrelfs/hehaorui/jiaqi/vc-dev/mls_en_transcripts.pkl"
                ):
                    # read MLS dataset transcript txt into dict
                    self.transcript_path = os.path.join(
                        self.dataset2dir["mls_train"].rstrip("audio/"),
                        "transcripts.txt",
                    )
                    file_bytes = self.client.get(self.transcript_path)
                    assert file_bytes is not None
                    buffer = io.BytesIO(file_bytes)
                    transcripts = buffer.getvalue()
                    del buffer
                    transcripts = transcripts.decode("utf-8")

                    # read MLS dataset transcript txt into dict
                    self.transcripts = {}
                    for line in transcripts.split("\n"):
                        if line == "":
                            continue
                        uid, transcript = line.split("\t")
                        self.transcripts[uid] = transcript

                    # dump cache
                    pickle.dump(self.transcripts, open("mls_en_transcripts.pkl", "wb"))
                self.transcripts = pickle.load(
                    open(
                        "/mnt/petrelfs/hehaorui/jiaqi/vc-dev/mls_en_transcripts.pkl",
                        "rb",
                    )
                )
            elif dataset_name == "librilight_medium":
                self.meta_data_cache_path = os.path.join(
                    dataset_cache_dir, f"{dataset_name}_metadata_cache.csv"
                )
                print(f"Loaded metadata cache from {self.meta_data_cache_path}")

                # write language info
                tmp_cache = pd.read_csv(self.meta_data_cache_path, encoding="utf-8")
                tmp_cache["language"] = "en"

                if self.meta_data_cache == None:
                    self.meta_data_cache = tmp_cache
                else:
                    self.meta_data_cache.append(tmp_cache)
                breakpoint()
                # TODO: load transcripts
                raise NotImplementedError

            elif dataset_name == "mls_german":
                raise NotImplementedError
                transcripts = pickle.load(
                    open(
                        "/mnt/petrelfs/hehaorui/jiaqi/gpt-tts/mls_german_transcripts.pkl",
                        "rb",
                    )
                )

        # set random_state to current time
        current_time = int(time.time())
        self.meta_data_cache = self.meta_data_cache.sample(
            frac=1.0, random_state=current_time
        ).reset_index(drop=True)

        # filter_by_length: filter_out files with duration < 3.0 or > 25.0
        print(f"Filtering files with duration between 3.0 and 25.0 seconds")
        print(f"Before filtering: {len(self.meta_data_cache)}")
        self.meta_data_cache = self.meta_data_cache[
            (self.meta_data_cache["duration"] >= 3.0)
            & (self.meta_data_cache["duration"] <= 25.0)
        ]
        print(f"After filtering: {len(self.meta_data_cache)}")
        # create speaker2speaker_id
        # self.speaker2id = self.create_speaker2id()
        self.all_num_frames = (self.meta_data_cache["duration"] * SAMPLE_RATE).to_list()
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )

    def save_cache_files(
        self,
        relpath2duration_path,
        relpath2speaker_path,
        index2relpath_path,
        relpath2duration,
        relpath2speaker,
        index2relpath,
    ):
        def safe_write_to_file(data, file_path, mode="w"):
            try:
                with lock, open(file_path, mode, encoding="utf-8") as f:
                    json.dump(data, f)
                    f.flush()
                    os.fsync(f.fileno())
            except IOError as e:
                print(f"Error writing to {file_path}: {e}")

        safe_write_to_file(relpath2duration, relpath2duration_path)
        print(f"Saved relpath2duration to {relpath2duration_path}")
        safe_write_to_file(relpath2speaker, relpath2speaker_path)
        print(f"Saved relpath2speaker to {relpath2speaker_path}")
        safe_write_to_file(index2relpath, index2relpath_path)
        print(f"Saved index2relpath to {index2relpath_path}")

    def create_metadata_cache(self, dataset, cache_dir):
        dataset_relpath2duration_path = os.path.join(
            cache_dir, f"{dataset}_relpath2duration.json"
        )
        dataset_relpath2speaker_path = os.path.join(
            cache_dir, f"{dataset}_relpath2speaker.json"
        )
        dataset_index2relpath_path = os.path.join(
            cache_dir, f"{dataset}_index2relpath.json"
        )
        dataset_meta_data_cache_path = os.path.join(
            cache_dir, f"{dataset}_metadata_cache.csv"
        )

        # if os.path.exists(dataset_relpath2duration_path) and os.path.exists(dataset_relpath2speaker_path) and os.path.exists(dataset_index2relpath_path):
        #     print(f"Loading cache for {dataset}")
        #     with open(dataset_relpath2duration_path, 'r', encoding='utf-8') as f:
        #         relpath2duration = json.load(f)
        #     with open(dataset_relpath2speaker_path, 'r', encoding='utf-8') as f:
        #         relpath2speaker = json.load(f)
        #     with open(dataset_index2relpath_path, 'r', encoding='utf-8') as f:
        #         index2relpath = json.load(f)
        #     print(f"Loaded cache for {dataset} with {len(relpath2duration)} files")
        # else:
        if True:
            print(f"Creating cache for {dataset}")
            relpath2duration = {}
            relpath2speaker = {}
            index2relpath = {}
            audio_rel_paths = self.get_audio_files(self.dataset2dir[dataset])
            random.shuffle(audio_rel_paths)
            print(f"Loaded {len(audio_rel_paths)} files from {dataset}")
            print(f"Generating cache for {dataset}")
            relpath2duration, relpath2speaker, index2relpath = (
                self.get_duration_speaker_and_filter(dataset, audio_rel_paths)
            )
            print(f"Generated cache for {dataset} with {len(relpath2duration)} files")
            print(f"Saving cache for {dataset}")
            self.save_cache_files(
                dataset_relpath2duration_path,
                dataset_relpath2speaker_path,
                dataset_index2relpath_path,
                relpath2duration,
                relpath2speaker,
                index2relpath,
            )
            print(f"Saved cache for {dataset}")

        meta_datas = []
        print(f"Generating metadata cache for {dataset}")
        for idx, relpath in tqdm(index2relpath.items()):
            temp_item = {
                "uid": f"{dataset}#{str(idx)}",
                "relpath": relpath,
                "duration": relpath2duration[relpath],
                "speaker": relpath2speaker[relpath],
            }
            meta_datas.append(temp_item)
        dataset_meta_data_cache = pd.DataFrame(meta_datas)
        dataset_meta_data_cache.to_csv(
            dataset_meta_data_cache_path, index=False, encoding="utf-8"
        )
        return dataset_meta_data_cache

    def get_duration_speaker_and_filter(self, dataset, audio_rel_paths):
        print(f"Processing metadata...")
        rel_path2duration = {}
        rel_path2speaker = {}
        idx2rel_path = {}
        base_dir = self.dataset2dir[dataset]
        full_paths = [os.path.join(base_dir, rel_path) for rel_path in audio_rel_paths]
        with Pool(processes=NUM_WORKERS) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(get_duration, full_paths),
                    total=len(audio_rel_paths),
                )
            )

        idx = 0
        print(f"Filtering files with duration between 3.0 and 25.0 seconds")
        for file, duration in tqdm(results):
            if duration > 3.0 and duration < 25.0:
                rel_path = os.path.relpath(file, base_dir)
                rel_path2duration[rel_path] = duration
                speaker_id = file.split(os.sep)[-3]
                speaker = f"{dataset}_{speaker_id}"
                rel_path2speaker[rel_path] = speaker
                idx2rel_path[idx] = rel_path
                idx += 1
        return rel_path2duration, rel_path2speaker, idx2rel_path

    def get_audio_files(self, directory):
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".flac", ".wav", ".opus")):
                    rel_path = os.path.relpath(os.path.join(root, file), directory)
                    audio_files.append(rel_path)
        return audio_files

    # only includes audio tokens
    def get_num_frames(self, index):
        # get_num_frames(durations) by index
        duration = self.meta_data_cache["duration"][index]
        # num_frames = duration * SAMPLE_RATE
        num_frames = int(duration * 50)

        # file_rel_path = self.meta_data_cache['relpath'][index]
        # uid = file_rel_path.rstrip('.flac').split('/')[-1]
        # num_frames += len(self.transcripts[uid])
        return num_frames

    def create_speaker2id(self):
        all_speakers = self.meta_data_cache["speaker"].unique()
        speaker2id = {}
        for idx, speaker in enumerate(all_speakers):
            speaker2id[speaker] = idx
        return speaker2id

    def snr_mixer(self, clean, noise, snr):
        # Normalizing to -25 dB FS
        rmsclean = (clean**2).mean() ** 0.5
        epsilon = 1e-10
        rmsclean = max(rmsclean, epsilon)
        scalarclean = 10 ** (-25 / 20) / rmsclean
        clean = clean * scalarclean

        rmsnoise = (noise**2).mean() ** 0.5
        rmsnoise = max(rmsnoise, epsilon)
        if rmsnoise == epsilon:
            return clean / scalarclean
        scalarnoise = 10 ** (-25 / 20) / rmsnoise
        noise = noise * scalarnoise
        rmsnoise = (noise**2).mean() ** 0.5

        # Set the noise level for a given SNR
        noisescalar = np.sqrt(rmsclean / (10 ** (snr / 20)) / rmsnoise)
        noisenewlevel = noise * noisescalar
        noisyspeech = clean + noisenewlevel
        noisyspeech_tensor = torch.tensor(noisyspeech, dtype=torch.float32)
        return noisyspeech_tensor

    def add_noise(self, clean):
        # self.noise_filenames: list of noise files
        random_idx = np.random.randint(0, np.size(self.noise_filenames))
        selected_noise_file = self.noise_filenames[random_idx]
        noise, _ = librosa.load(selected_noise_file, sr=SAMPLE_RATE)
        clean = clean.cpu().numpy()
        if len(noise) >= len(clean):
            noise = noise[0 : len(clean)]  # 截取噪声的长度
        else:
            while len(noise) <= len(clean):  # 如果噪声的长度小于语音的长度
                random_idx = (random_idx + 1) % len(
                    self.noise_filenames
                )  # 随机读一个噪声
                newnoise, fs = librosa.load(selected_noise_file, sr=SAMPLE_RATE)
                noiseconcat = np.append(
                    noise, np.zeros(int(fs * 0.2))
                )  # 在噪声后面加上0.2静音
                noise = np.append(noiseconcat, newnoise)  # 拼接噪声
        noise = noise[0 : len(clean)]  # 截取噪声的长度
        # 随机sample一个小于20大于0的随机数
        snr = random.uniform(0.0, 15.0)
        noisyspeech = self.snr_mixer(
            clean=clean, noise=noise, snr=snr
        )  # 根据随机的SNR级别，混合生成带噪音频
        del noise
        return noisyspeech

    def add_reverb(self, speech):
        room_dim = [
            np.random.uniform(1, 12) for _ in range(3)
        ]  # [length, width, height]
        mic_pos = [np.random.uniform(0, dim) for dim in room_dim]  # 随机选择麦克风位置
        distance = np.random.normal(2, 4)  # 确定声源与麦克风的距离
        while distance <= 0 or distance > 5:
            distance = np.random.normal(2, 4)
        source_pos = [
            mic_pos[0] + distance,
            mic_pos[1],
            mic_pos[2],
        ]  # 随机选择声源位置，确保它在以麦克风为中心的球内
        rt60 = np.random.uniform(0.05, 1.0)  # 随机选择RT60值
        try:
            rir_filter = rir.generate(
                c=340,  # 声速
                fs=SAMPLE_RATE,
                r=[mic_pos],  # 麦克风位置
                s=source_pos,  # 声源位置
                L=room_dim,  # 房间尺寸
                reverberation_time=rt60,  # RT60值
                nsample=4096,  # IR长度
            )
            # 应用混响
            speech_reverb = np.convolve(
                speech.cpu().numpy(), rir_filter[:, 0], mode="same"
            )
            speech = torch.tensor(speech_reverb, dtype=torch.float32)
            return speech
        except:
            return speech  # 如果遇到ValueError: s is outside the room，直接返回没加混响的声音

    def __len__(self):
        return len(self.meta_data_cache)

    def __getitem__(self, idx):
        # Get the file rel path
        file_rel_path = self.meta_data_cache["relpath"][idx]
        # Get the dataset from cache uid
        dataset_name = self.meta_data_cache["uid"][idx].split("#")[0]
        # Get the full file path
        full_file_path = os.path.join(self.dataset2dir[dataset_name], file_rel_path)

        # get transcript
        uid = file_rel_path.rstrip(".flac").split("/")[-1]
        phone = self.transcripts[uid]
        phone = phonemizer_g2p(phone, "en")[1]
        # phone = [LANG2CODE['en']] + phone
        # phone = torch.tensor(phone, dtype=torch.long)

        file_bytes = self.client.get(full_file_path)
        assert file_bytes is not None, f"file {full_file_path} not found"
        buffer = io.BytesIO(file_bytes)
        speech, _ = librosa.load(buffer, sr=SAMPLE_RATE)
        speech = torch.tensor(speech, dtype=torch.float32)
        # pad speech to multiples of 320
        remainder = speech.size(0) % 320
        if remainder > 0:
            pad = 320 - remainder
            speech = torch.cat([speech, torch.zeros(pad, dtype=torch.float32)], dim=0)

        # inputs = self._get_reference_vc(speech, hop_length=200)
        inputs = {}
        # Get the speaker id
        # speaker = self.meta_data_cache['speaker'][idx]
        # speaker_id = self.speaker2id[speaker]
        # inputs["speaker_id"] = speaker_id
        inputs["speech"] = speech  # 24khz speech, [T]
        inputs["phone"] = phone  # [T]
        return inputs


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
