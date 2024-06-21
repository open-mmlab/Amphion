# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from tqdm import tqdm
from g2p_en import G2p
import librosa
from torch.utils.data import Dataset
import pandas as pd
import time
import io

SAMPLE_RATE = 16000
# g2p
from .g2p_processor import G2pProcessor

phonemizer_g2p = G2pProcessor()


class VALLEDataset(Dataset):
    def __init__(self, args):
        print(f"Initializing VALLEDataset")
        self.dataset_list = args.dataset_list

        print(f"using sampling rate {SAMPLE_RATE}")

        # set dataframe clumn name
        book_col_name = [
            "ID",
            "Original_text",
            "Normalized_text",
            "Aligned_or_not",
            "Start_time",
            "End_time",
            "Signal_to_noise_ratio",
        ]
        trans_col_name = [
            "ID",
            "Original_text",
            "Normalized_text",
            "Dir_path",
            "Duration",
        ]
        self.metadata_cache = pd.DataFrame(columns=book_col_name)
        self.trans_cache = pd.DataFrame(columns=trans_col_name)
        # dataset_cache_dir = args.cache_dir # cache_dir
        # print(f"args.cache_dir = ", args.cache_dir)
        # os.makedirs(dataset_cache_dir, exist_ok=True)

        ######## add data dir to dataset2dir ##########
        self.dataset2dir = {
            "dev-clean": f"{args.data_dir}/dev-clean",
            "dev-other": f"{args.data_dir}/dev-other",
            "test-clean": f"{args.data_dir}/test-clean",
            "test-other": f"{args.data_dir}/test-other",
            "train-clean-100": f"{args.data_dir}/train-clean-100",
            "train-clean-360": f"{args.data_dir}/train-clean-360",
            "train-other-500": f"{args.data_dir}/train-other-500",
        }

        ###### load metadata and transcripts #####
        for dataset_name in self.dataset_list:
            print("Initializing dataset: ", dataset_name)
            # get [book,transcripts,audio] files list
            self.book_files_list = self.get_metadata_files(
                self.dataset2dir[dataset_name]
            )
            self.trans_files_list = self.get_trans_files(self.dataset2dir[dataset_name])

            ## create metadata_cache (book.tsv file is not filtered, some file is not exist, but contain Duration and Signal_to_noise_ratio)
            print("reading paths for dataset...")
            for book_path in tqdm(self.book_files_list):
                tmp_cache = pd.read_csv(
                    book_path, sep="\t", names=book_col_name, quoting=3
                )
                self.metadata_cache = pd.concat(
                    [self.metadata_cache, tmp_cache], ignore_index=True
                )
            self.metadata_cache.set_index("ID", inplace=True)

            ## create transcripts (the trans.tsv file)
            print("creating transcripts for dataset...")
            for trans_path in tqdm(self.trans_files_list):
                tmp_cache = pd.read_csv(
                    trans_path, sep="\t", names=trans_col_name, quoting=3
                )
                tmp_cache["Dir_path"] = os.path.dirname(trans_path)
                self.trans_cache = pd.concat(
                    [self.trans_cache, tmp_cache], ignore_index=True
                )
            self.trans_cache.set_index("ID", inplace=True)

            ## calc duration
            self.trans_cache["Duration"] = (
                self.metadata_cache.End_time[self.trans_cache.index]
                - self.metadata_cache.Start_time[self.trans_cache.index]
            )
            ## add fullpath
            # self.trans_cache['Full_path'] = os.path.join(self.dataset2dir[dataset_name],self.trans_cache['ID'])

        # filter_by_duration: filter_out files with duration < 3.0 or > 15.0
        print(f"Filtering files with duration between 3.0 and 15.0 seconds")
        print(f"Before filtering: {len(self.trans_cache)}")
        self.trans_cache = self.trans_cache[
            (self.trans_cache["Duration"] >= 3.0)
            & (self.trans_cache["Duration"] <= 15.0)
        ]
        print(f"After filtering: {len(self.trans_cache)}")

    def get_metadata_files(self, directory):
        book_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".book.tsv") and file[0] != ".":
                    rel_path = os.path.join(root, file)
                    book_files.append(rel_path)
        return book_files

    def get_trans_files(self, directory):
        trans_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".trans.tsv") and file[0] != ".":
                    rel_path = os.path.join(root, file)
                    trans_files.append(rel_path)
        return trans_files

    def get_audio_files(self, directory):
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".flac", ".wav", ".opus")):
                    rel_path = os.path.relpath(os.path.join(root, file), directory)
                    audio_files.append(rel_path)
        return audio_files

    def get_num_frames(self, index):
        # get_num_frames(durations) by index
        duration = self.meta_data_cache["Duration"][index]
        # num_frames = duration * SAMPLE_RATE
        num_frames = int(duration * 75)

        # file_rel_path = self.meta_data_cache['relpath'][index]
        # uid = file_rel_path.rstrip('.flac').split('/')[-1]
        # num_frames += len(self.transcripts[uid])
        return num_frames

    def __len__(self):
        return len(self.trans_cache)

    def __getitem__(self, idx):
        # Get the file rel path
        file_dir_path = self.trans_cache["Dir_path"].iloc[idx]
        # Get uid
        uid = self.trans_cache.index[idx]
        # Get the file name from cache uid
        file_name = uid + ".wav"
        # Get the full file path
        full_file_path = os.path.join(file_dir_path, file_name)

        # get phone
        phone = self.trans_cache["Normalized_text"][uid]
        phone = phonemizer_g2p(phone, "en")[1]
        # load speech
        speech, _ = librosa.load(full_file_path, sr=SAMPLE_RATE)
        # if self.resample_to_24k:
        #     speech = librosa.resample(speech, orig_sr=SAMPLE_RATE, target_sr=24000)
        # speech = torch.tensor(speech, dtype=torch.float32)
        # pad speech to multiples of 200

        # remainder = speech.size(0) % 200
        # if remainder > 0:
        #     pad = 200 - remainder
        #     speech = torch.cat([speech, torch.zeros(pad, dtype=torch.float32)], dim=0)

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


def test():
    from utils.util import load_config

    cfg = load_config("./egs/tts/VALLE_V2/exp_ar_libritts.json")
    dataset = VALLEDataset(cfg.dataset)
    metadata_cache = dataset.metadata_cache
    trans_cache = dataset.trans_cache
    print(trans_cache.head(10))
    # print(dataset.book_files_list)
    breakpoint()


if __name__ == "__main__":
    test()
