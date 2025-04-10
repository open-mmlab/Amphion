# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random

from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)


class ScheduledSampler(Sampler):
    """A sampler that samples data from a given concat-dataset.

    Args:
        concat_dataset (ConcatDataset): a concatenated dataset consisting of all datasets
        batch_size (int): batch size
        holistic_shuffle (bool): whether to shuffle the whole dataset or not
        logger (logging.Logger): logger to print warning message

    Usage:
        For cfg.train.batch_size = 3, cfg.train.holistic_shuffle = False, cfg.train.drop_last = True:
        >>> list(ScheduledSampler(ConcatDataset([[0, 1, 2], [3, 4, 5], [6, 7, 8]])))
        [3, 4, 5, 0, 1, 2, 6, 7, 8]
    """

    def __init__(
        self,
        concat_dataset,
        batch_size,
        holistic_shuffle,
        logger=None,
        loader_type="train",
    ):
        if not isinstance(concat_dataset, ConcatDataset):
            raise ValueError(
                "concat_dataset must be an instance of ConcatDataset, but got {}".format(
                    type(concat_dataset)
                )
            )
        if not isinstance(batch_size, int):
            raise ValueError(
                "batch_size must be an integer, but got {}".format(type(batch_size))
            )
        if not isinstance(holistic_shuffle, bool):
            raise ValueError(
                "holistic_shuffle must be a boolean, but got {}".format(
                    type(holistic_shuffle)
                )
            )

        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.holistic_shuffle = holistic_shuffle

        affected_dataset_name = []
        affected_dataset_len = []
        for dataset in concat_dataset.datasets:
            dataset_len = len(dataset)
            dataset_name = dataset.get_dataset_name()
            if dataset_len < batch_size:
                affected_dataset_name.append(dataset_name)
                affected_dataset_len.append(dataset_len)

        self.type = loader_type
        for dataset_name, dataset_len in zip(
            affected_dataset_name, affected_dataset_len
        ):
            if not loader_type == "valid":
                logger.warning(
                    "The {} dataset {} has a length of {}, which is smaller than the batch size {}. This may cause unexpected behavior.".format(
                        loader_type, dataset_name, dataset_len, batch_size
                    )
                )

    def __len__(self):
        # the number of batches with drop last
        num_of_batches = sum(
            [
                math.floor(len(dataset) / self.batch_size)
                for dataset in self.concat_dataset.datasets
            ]
        )
        # if samples are not enough for one batch, we don't drop last
        if self.type == "valid" and num_of_batches < 1:
            return len(self.concat_dataset)
        return num_of_batches * self.batch_size

    def __iter__(self):
        iters = []
        for dataset in self.concat_dataset.datasets:
            iters.append(
                SequentialSampler(dataset).__iter__()
                if not self.holistic_shuffle
                else RandomSampler(dataset).__iter__()
            )
        # e.g. [0, 200, 400]
        init_indices = [0] + self.concat_dataset.cumulative_sizes[:-1]
        output_batches = []
        for dataset_idx in range(len(self.concat_dataset.datasets)):
            cur_batch = []
            for idx in iters[dataset_idx]:
                cur_batch.append(idx + init_indices[dataset_idx])
                if len(cur_batch) == self.batch_size:
                    output_batches.append(cur_batch)
                    cur_batch = []
            # if loader_type is valid, we don't need to drop last
            if self.type == "valid" and len(cur_batch) > 0:
                output_batches.append(cur_batch)

        # force drop last in training
        random.shuffle(output_batches)
        output_indices = [item for sublist in output_batches for item in sublist]
        return iter(output_indices)


def build_samplers(concat_dataset: Dataset, cfg, logger, loader_type):
    sampler = ScheduledSampler(
        concat_dataset,
        cfg.train.batch_size,
        cfg.train.sampler.holistic_shuffle,
        logger,
        loader_type,
    )
    batch_sampler = BatchSampler(
        sampler,
        cfg.train.batch_size,
        cfg.train.sampler.drop_last if not loader_type == "valid" else False,
    )
    return sampler, batch_sampler


class VariableSampler(BatchSampler):
    def __init__(self, batches, drop_last: bool, use_random_sampler=False):
        self.data_list = batches
        if use_random_sampler:
            self.sampler = RandomSampler(batches)
        else:
            self.sampler = SequentialSampler(batches)

        self.start_index = 0

        super().__init__(self.sampler, 1, drop_last)

    def skip_steps(self, steps):
        """Skip a specified number of steps

        Args:
            steps (int): The number of steps to skip
        """
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        self.start_index = steps % len(self.data_list)

    def __iter__(self):
        for i in range(self.start_index, len(self.data_list)):
            yield self.data_list[i]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
