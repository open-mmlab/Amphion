# This module is from [WeNet](https://github.com/wenet-e2e/wenet).

# ## Citations

# ```bibtex
# @inproceedings{yao2021wenet,
#   title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
#   author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
#   booktitle={Proc. Interspeech},
#   year={2021},
#   address={Brno, Czech Republic },
#   organization={IEEE}
# }

# @article{zhang2022wenet,
#   title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
#   author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
#   journal={arXiv preprint arXiv:2203.15455},
#   year={2022}
# }
#

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import wenet.dataset.processor as processor
from wenet.utils.file_utils import read_lists


class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """Return an iterator over the source dataset processed by the
        given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """Sample data according to rank/world_size/num_workers

        Args:
            data(List): input data list

        Returns:
            List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank :: self.world_size]
        data = data[self.worker_id :: self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def Dataset(
    data_type,
    data_list_file,
    symbol_table,
    conf,
    bpe_model=None,
    non_lang_syms=None,
    partition=True,
):
    """Construct dataset from arguments

    We have two shuffle stage in the Dataset. The first is global
    shuffle at shards tar/raw file level. The second is global shuffle
    at training samples level.

    Args:
        data_type(str): raw/shard
        bpe_model(str): model for english bpe part
        partition(bool): whether to do data partition in terms of rank
    """
    assert data_type in ["raw", "shard"]
    lists = read_lists(data_list_file)
    shuffle = conf.get("shuffle", True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    if data_type == "shard":
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    else:
        dataset = Processor(dataset, processor.parse_raw)

    dataset = Processor(
        dataset,
        processor.tokenize,
        symbol_table,
        bpe_model,
        non_lang_syms,
        conf.get("split_with_space", False),
    )
    filter_conf = conf.get("filter_conf", {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    resample_conf = conf.get("resample_conf", {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    speed_perturb = conf.get("speed_perturb", False)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)

    feats_type = conf.get("feats_type", "fbank")
    assert feats_type in ["fbank", "mfcc"]
    if feats_type == "fbank":
        fbank_conf = conf.get("fbank_conf", {})
        dataset = Processor(dataset, processor.compute_fbank, **fbank_conf)
    elif feats_type == "mfcc":
        mfcc_conf = conf.get("mfcc_conf", {})
        dataset = Processor(dataset, processor.compute_mfcc, **mfcc_conf)

    spec_aug = conf.get("spec_aug", True)
    spec_sub = conf.get("spec_sub", False)
    spec_trim = conf.get("spec_trim", False)
    if spec_aug:
        spec_aug_conf = conf.get("spec_aug_conf", {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)
    if spec_sub:
        spec_sub_conf = conf.get("spec_sub_conf", {})
        dataset = Processor(dataset, processor.spec_sub, **spec_sub_conf)
    if spec_trim:
        spec_trim_conf = conf.get("spec_trim_conf", {})
        dataset = Processor(dataset, processor.spec_trim, **spec_trim_conf)

    if shuffle:
        shuffle_conf = conf.get("shuffle_conf", {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get("sort", True)
    if sort:
        sort_conf = conf.get("sort_conf", {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    batch_conf = conf.get("batch_conf", {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset
