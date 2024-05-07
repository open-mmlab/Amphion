from .mel_processing import spectrogram_torch
from .commons import rand_spec_segments, slice_segments

import os
import random
import torch
from torch.utils.data import Dataset, Sampler
import torchaudio


def read_txt_lines(path):
    ans = []
    with open(str(path), "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                ans.append(line)
    return ans


class FreeVCDataset(Dataset):
    def __init__(self, audiopaths, hparams):
        self.audiopaths = read_txt_lines(audiopaths)

        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.use_spk = hparams.model.use_spk
        self.spec_len = hparams.train.max_speclen

        self.vctk_16k_dir = hparams.preprocess.vctk_16k_dir
        self.spk_dir = hparams.preprocess.spk_dir
        self.ssl_dir = hparams.preprocess.ssl_dir
        self.sr_dir = hparams.preprocess.sr_dir

        random.shuffle(self.audiopaths)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        lengths = []
        for audiopath in self.audiopaths:
            path = os.path.join(self.vctk_16k_dir, audiopath)
            lengths.append(os.path.getsize(path) // (2 * self.hop_length))
        self.lengths = lengths

    @torch.no_grad()
    def load_sample(self, filename):
        filepath = os.path.join(self.vctk_16k_dir, filename)
        audio, sampling_rate = torchaudio.load(filepath)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )

        spec = spectrogram_torch(
            audio,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        ).squeeze_(0)

        if self.use_spk:
            spk_path = os.path.join(self.spk_dir, filename.replace(".wav", ".pt"))
            spk = torch.load(spk_path)
        else:
            spk = None

        if not self.use_sr:
            ssl_path = os.path.join(self.ssl_dir, filename.replace(".wav", ".pt"))
            ssl = torch.load(ssl_path).squeeze_(0)
        else:
            h = random.randint(68, 92)
            ssl_path = os.path.join(self.sr_dir, filename.replace(".wav", f"_{h}.pt"))
            ssl = torch.load(ssl_path).squeeze_(0)

        return ssl, spec, audio, spk

    def __getitem__(self, index):
        return self.load_sample(self.audiopaths[index])

    def __len__(self):
        return len(self.audiopaths)


class FreeVCCollate:
    def __init__(self, hps):
        self.hps = hps
        self.use_sr = hps.train.use_sr
        self.use_spk = hps.model.use_spk

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if self.use_spk:
            spks = torch.FloatTensor(len(batch), batch[0][3].size(0))
        else:
            spks = None

        c_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, : c.size(1)] = c

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            if self.use_spk:
                spks[i] = row[3]  # type: ignore

        spec_seglen = (
            spec_lengths[-1]
            if spec_lengths[-1] < self.hps.train.max_speclen + 1
            else self.hps.train.max_speclen + 1
        )
        wav_seglen = spec_seglen * self.hps.data.hop_length

        spec_padded, ids_slice = rand_spec_segments(
            spec_padded,
            spec_lengths,
            spec_seglen,  # type: ignore
        )
        wav_padded = slice_segments(
            wav_padded, ids_slice * self.hps.data.hop_length, wav_seglen
        )

        c_padded = slice_segments(c_padded, ids_slice, spec_seglen)[:, :, :-1]  # type: ignore

        spec_padded = spec_padded[:, :, :-1]
        wav_padded = wav_padded[:, :, : -self.hps.data.hop_length]

        return c_padded, spec_padded, wav_padded, spks


class BucketSampler(Sampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        shuffle=True,
    ):
        super().__init__(dataset)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.num_replicas = 1

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

        self.shuffle = shuffle

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # # deterministically shuffle based on epoch
        # g = torch.Generator()
        # g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket)).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # # subsample
            # ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
