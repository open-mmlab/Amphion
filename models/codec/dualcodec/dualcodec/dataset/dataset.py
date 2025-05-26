# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Amphion. All Rights Reserved
#
################################################################################

import random
import json
import math
from functools import partial
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from .file_utils import read_lists, read_json_lists


class Processor(IterableDataset):
    """
    Processor IterableDataset
    """

    def __init__(self, source, f, *args, **kw):
        """
        :param source:
        :param f:
        :param args:
        :param kw:
        """
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        """
        :param epoch:
        :return:
        """
        self.source.set_epoch(epoch)

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        """Return an iterator over the source dataset processed by the
        given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        """
        :param f:
        :return:
        """
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    """
    DistributedSampler
    """

    def __init__(self, shuffle=True, partition=True):
        """
        :param shuffle:
        :param partition:
        """
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        """
        :return:
        """
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            print(
                "WARNING: distributed not initialized in DistributedSampler! \
                  Defaulting to single rank. \
                  Or set `manual_dist_sampler` to True in `gluster_opener`."
            )
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            print("no dataloader worker info!")
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        print("DistributedSampler:")
        ret = dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )
        print(ret)
        return ret

    def set_epoch(self, epoch):
        """
        :param epoch:
        :return:
        """
        self.epoch = epoch

    def sample(self, data):
        """Sample data according to rank/world_size/num_workers

        Args:
            data(List): input data list

        Returns:
            List: data list after sample
        """
        data = list(range(len(data)))
        # force datalist even
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))
                data = data[: self.world_size]
            data = data[self.rank :: self.world_size]
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))
            data = data[: self.num_workers]
        data = data[self.worker_id :: self.num_workers]
        print("data idx: ", data[:10])
        print("data length:", len(data))

        return data


class DataList(IterableDataset):
    """
    DataList IterableDataset
    """

    def __init__(self, lists, shuffle=True, partition=True):
        """
        :param lists:
        :param shuffle:
        :param partition:
        """
        self.lists = lists
        self.partition = partition
        if partition:
            self.sampler = DistributedSampler(shuffle, partition)
            self.sampler_info = self.sampler.update()
            self.indexes = self.sampler.sample(self.lists)
        else:
            pass

    def set_epoch(self, epoch):
        """
        :param epoch:
        :return:
        """
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.indexes)

    def __iter__(self):
        """
        :return:
        """
        if self.partition:
            for index in self.indexes:
                try:
                    data = dict(src=self.lists[index])
                    data.update(self.sampler_info)
                    yield data
                except Exception as e:
                    print(e)
                    continue
        else:
            for i in self.lists:
                data = dict(src=i)
                yield data


def Dataset(
    data_list_file,
    data_pipeline,
    mode="train",
    shuffle=True,
    partition=True,
    tts_file="",
    prompt_utt2data="",
):
    """Construct dataset from arguments

    We have two shuffle stage in the Dataset. The first is global
    shuffle at shards tar/raw file level. The second is global shuffle
    at training samples level.

    Args:
        data_type(str): raw/shard
        tokenizer (BaseTokenizer): tokenizer to tokenize
        partition(bool): whether to do data partition in terms of rank
    """
    assert mode in ["train", "inference"]
    lists = read_lists(data_list_file)
    if mode == "inference":
        with open(tts_file) as f:
            tts_data = json.load(f)
        utt2lists = read_json_lists(prompt_utt2data)
        # filter unnecessary file in inference mode
        lists = list(
            set([utt2lists[utt] for utt in tts_data.keys() if utt2lists[utt] in lists])
        )
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    if mode == "inference":
        # map partial arg tts_data in inference mode
        data_pipeline[0] = partial(data_pipeline[0], tts_data=tts_data)
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)
    return dataset


def prepare_gluster_dataset(data_list: DataList, data_pipeline: list, mode="train"):
    """
    Args:
        data_list (DataList):
        data_pipeline (list):
        mode (str):
    Returns:
    """
    for func in data_pipeline:
        data_list = Processor(data_list, func, mode=mode)
    return data_list
