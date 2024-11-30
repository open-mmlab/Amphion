# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from multiprocessing import Pool, Lock
import random
import torchaudio


NUM_WORKERS = 64
lock = Lock()
SAMPLE_RATE = 16000


def get_metadata(file_path):
    metadata = torchaudio.info(file_path)
    return file_path, metadata.num_frames


def get_speaker(file_path):
    speaker_id = file_path.split(os.sep)[-3]
    if "mls" in file_path:
        speaker = "mls_" + speaker_id
    else:
        speaker = "libri_" + speaker_id
    return file_path, speaker


def safe_write_to_file(data, file_path, mode="w"):
    try:
        with lock, open(file_path, mode, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
    except IOError as e:
        print(f"Error writing to {file_path}: {e}")


class VCDataset(Dataset):
    def __init__(self, args, TRAIN_MODE=True):
        print(f"Initializing VCDataset")
        if TRAIN_MODE:
            directory_list = args.directory_list
        else:
            directory_list = args.test_directory_list
        random.shuffle(directory_list)

        self.use_ref_noise = args.use_ref_noise
        print(f"use_ref_noise: {self.use_ref_noise}")

        # number of workers
        print(f"Using {NUM_WORKERS} workers")
        self.directory_list = directory_list
        print(f"Loading {len(directory_list)} directories: {directory_list}")

        self.metadata_cache = {}
        self.speaker_cache = {}

        self.files = []
        # Load all flac files
        for directory in directory_list:
            print(f"Loading {directory}")
            files = self.get_flac_files(directory)
            random.shuffle(files)
            print(f"Loaded {len(files)} files")
            self.files.extend(files)
            del files
            print(f"Now {len(self.files)} files")
            self.meta_data_cache = self.process_files()
            self.speaker_cache = self.process_speakers()

        print(f"Loaded {len(self.files)} files")
        random.shuffle(self.files)  # Shuffle the files.

        self.filtered_files, self.all_num_frames, index2numframes, index2speakerid = (
            self.filter_files()
        )
        print(f"Loaded {len(self.filtered_files)} files")

        self.index2numframes = index2numframes
        self.index2speaker = index2speakerid
        self.speaker2id = self.create_speaker2id()
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )
        del self.meta_data_cache, self.speaker_cache

        if self.use_ref_noise:
            if TRAIN_MODE:
                self.noise_filenames = self.get_all_flac(args.noise_dir)
            else:
                self.noise_filenames = self.get_all_flac(args.test_noise_dir)

    def process_files(self):
        print(f"Processing metadata...")
        files_to_process = [
            file for file in self.files if file not in self.metadata_cache
        ]
        if files_to_process:
            with Pool(processes=NUM_WORKERS) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(get_metadata, files_to_process),
                        total=len(files_to_process),
                    )
                )
            for file, num_frames in results:
                self.metadata_cache[file] = num_frames
        else:
            print(
                f"Skipping processing metadata, loaded {len(self.metadata_cache)} files"
            )
        return self.metadata_cache

    def process_speakers(self):
        print(f"Processing speakers...")
        files_to_process = [
            file for file in self.files if file not in self.speaker_cache
        ]
        if files_to_process:
            with Pool(processes=NUM_WORKERS) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(get_speaker, files_to_process),
                        total=len(files_to_process),
                    )
                )
            for file, speaker in results:
                self.speaker_cache[file] = speaker
        else:
            print(
                f"Skipping processing speakers, loaded {len(self.speaker_cache)} files"
            )
        return self.speaker_cache

    def get_flac_files(self, directory):
        flac_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                # flac or wav
                if file.endswith(".flac") or file.endswith(".wav"):
                    flac_files.append(os.path.join(root, file))
        return flac_files

    def get_all_flac(self, directory):
        directories = [
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        if not directories:
            return self.get_flac_files(directory)
        with Pool(processes=NUM_WORKERS) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(self.get_flac_files, directories),
                total=len(directories),
                desc="Processing",
            ):
                results.extend(result)
        print(f"Found {len(results)} waveform files")
        return results

    def get_num_frames(self, index):
        return self.index2numframes[index]

    def filter_files(self):
        # Filter files
        metadata_cache = self.meta_data_cache
        speaker_cache = self.speaker_cache
        filtered_files = []
        all_num_frames = []
        index2numframes = {}
        index2speaker = {}
        for file in self.files:
            num_frames = metadata_cache[file]
            if SAMPLE_RATE * 3 <= num_frames <= SAMPLE_RATE * 30:
                filtered_files.append(file)
                all_num_frames.append(num_frames)
                index2speaker[len(filtered_files) - 1] = speaker_cache[file]
                index2numframes[len(filtered_files) - 1] = num_frames
        return filtered_files, all_num_frames, index2numframes, index2speaker

    def create_speaker2id(self):
        speaker2id = {}
        unique_id = 0
        print(f"Creating speaker2id from {len(self.index2speaker)} utterences")
        for _, speaker in tqdm(self.index2speaker.items()):
            if speaker not in speaker2id:
                speaker2id[speaker] = unique_id
                unique_id += 1
        print(f"Created speaker2id with {len(speaker2id)} speakers")
        return speaker2id

    def snr_mixer(self, clean, noise, snr):
        # Normalizing to -25 dB FS
        rmsclean = (clean**2).mean() ** 0.5
        epsilon = 1e-10
        rmsclean = max(rmsclean, epsilon)
        scalarclean = 10 ** (-25 / 20) / rmsclean
        clean = clean * scalarclean

        rmsnoise = (noise**2).mean() ** 0.5
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
        noise, _ = librosa.load(self.noise_filenames[random_idx], sr=SAMPLE_RATE)
        clean = clean.cpu().numpy()
        if len(noise) >= len(clean):
            noise = noise[0 : len(clean)]
        else:
            while len(noise) <= len(clean):
                random_idx = (random_idx + 1) % len(self.noise_filenames)
                newnoise, fs = librosa.load(
                    self.noise_filenames[random_idx], sr=SAMPLE_RATE
                )
                noiseconcat = np.append(noise, np.zeros(int(fs * 0.2)))
                noise = np.append(noiseconcat, newnoise)
        noise = noise[0 : len(clean)]
        snr = random.uniform(0.0, 20.0)
        noisyspeech = self.snr_mixer(clean=clean, noise=noise, snr=snr)
        del noise
        return noisyspeech

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.filtered_files[idx]
        speech, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(speech) > 30 * SAMPLE_RATE:
            speech = speech[: 30 * SAMPLE_RATE]
        speech = torch.tensor(speech, dtype=torch.float32)
        inputs = self._get_reference_vc(speech, hop_length=200)
        speaker = self.index2speaker[idx]
        speaker_id = self.speaker2id[speaker]
        inputs["speaker_id"] = speaker_id
        return inputs

    def _get_reference_vc(self, speech, hop_length):
        pad_size = 1600 - speech.shape[0] % 1600
        speech = torch.nn.functional.pad(speech, (0, pad_size))

        # hop_size
        frame_nums = speech.shape[0] // hop_length
        clip_frame_nums = np.random.randint(
            int(frame_nums * 0.25), int(frame_nums * 0.45)
        )
        clip_frame_nums += (frame_nums - clip_frame_nums) % 8
        start_frames, end_frames = 0, clip_frame_nums

        ref_speech = speech[start_frames * hop_length : end_frames * hop_length]
        new_speech = torch.cat(
            (speech[: start_frames * hop_length], speech[end_frames * hop_length :]), 0
        )

        ref_mask = torch.ones(len(ref_speech) // hop_length)
        mask = torch.ones(len(new_speech) // hop_length)
        if not self.use_ref_noise:
            # not use noise
            return {
                "speech": new_speech,
                "ref_speech": ref_speech,
                "ref_mask": ref_mask,
                "mask": mask,
            }
        else:
            # use reference noise
            noisy_ref_speech = self.add_noise(ref_speech)
            return {
                "speech": new_speech,
                "ref_speech": ref_speech,
                "noisy_ref_speech": noisy_ref_speech,
                "ref_mask": ref_mask,
                "mask": mask,
            }


class BaseCollator(object):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, T, n_mels]
        # frame_pitch, frame_energy: [1, T]
        # target_len: [1]
        # spk_id: [b, 1]
        # mask: [b, T, 1]

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "phone_len":
                packed_batch_features["phone_len"] = torch.LongTensor(
                    [b["phone_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["phone_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["phn_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "audio_len":
                packed_batch_features["audio_len"] = torch.LongTensor(
                    [b["audio_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["audio_len"], 1), dtype=torch.long) for b in batch
                ]
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )
        return packed_batch_features


class VCCollator(BaseCollator):
    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)
        self.use_ref_noise = self.cfg.trans_exp.use_ref_noise
        print(f"use_ref_noise: {self.use_ref_noise}")

    def __call__(self, batch):
        packed_batch_features = dict()

        # Function to handle tensor copying
        def process_tensor(data, dtype=torch.float32):
            if isinstance(data, torch.Tensor):
                return data.clone().detach()
            else:
                return torch.tensor(data, dtype=dtype)

        # Process 'speech' data
        speeches = [process_tensor(b["speech"]) for b in batch]
        packed_batch_features["speech"] = pad_sequence(
            speeches, batch_first=True, padding_value=0
        )

        # Process 'ref_speech' data
        ref_speeches = [process_tensor(b["ref_speech"]) for b in batch]
        packed_batch_features["ref_speech"] = pad_sequence(
            ref_speeches, batch_first=True, padding_value=0
        )

        # Process 'mask' data
        masks = [process_tensor(b["mask"]) for b in batch]
        packed_batch_features["mask"] = pad_sequence(
            masks, batch_first=True, padding_value=0
        )

        # Process 'ref_mask' data
        ref_masks = [process_tensor(b["ref_mask"]) for b in batch]
        packed_batch_features["ref_mask"] = pad_sequence(
            ref_masks, batch_first=True, padding_value=0
        )

        # Process 'speaker_id' data
        speaker_ids = [
            process_tensor(b["speaker_id"], dtype=torch.int64) for b in batch
        ]
        packed_batch_features["speaker_id"] = torch.stack(speaker_ids, dim=0)
        if self.use_ref_noise:
            # Process 'noisy_ref_speech' data
            noisy_ref_speeches = [process_tensor(b["noisy_ref_speech"]) for b in batch]
            packed_batch_features["noisy_ref_speech"] = pad_sequence(
                noisy_ref_speeches, batch_first=True, padding_value=0
            )
        return packed_batch_features


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
