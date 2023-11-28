# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.base.base_sampler import BatchSampler
from utils.util import (
    Logger,
    remove_older_ckpt,
    save_config,
    set_all_random_seed,
    ValueWindow,
)


class BaseTrainer(object):
    def __init__(self, args, cfg):
        self.args = args
        self.log_dir = args.log_dir
        self.cfg = cfg

        self.checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if not cfg.train.ddp or args.local_rank == 0:
            self.sw = SummaryWriter(os.path.join(args.log_dir, "events"))
            self.logger = self.build_logger()
        self.time_window = ValueWindow(50)

        self.step = 0
        self.epoch = -1
        self.max_epochs = self.cfg.train.epochs
        self.max_steps = self.cfg.train.max_steps

        # set random seed & init distributed training
        set_all_random_seed(self.cfg.train.random_seed)
        if cfg.train.ddp:
            dist.init_process_group(backend="nccl")

        if cfg.model_type not in ["AutoencoderKL", "AudioLDM"]:
            self.singers = self.build_singers_lut()

        # setup data_loader
        self.data_loader = self.build_data_loader()

        # setup model & enable distributed training
        self.model = self.build_model()
        print(self.model)

        if isinstance(self.model, dict):
            for key, value in self.model.items():
                value.cuda(self.args.local_rank)
                if key == "PQMF":
                    continue
                if cfg.train.ddp:
                    self.model[key] = DistributedDataParallel(
                        value, device_ids=[self.args.local_rank]
                    )
        else:
            self.model.cuda(self.args.local_rank)
            if cfg.train.ddp:
                self.model = DistributedDataParallel(
                    self.model, device_ids=[self.args.local_rank]
                )

        # create criterion
        self.criterion = self.build_criterion()
        if isinstance(self.criterion, dict):
            for key, value in self.criterion.items():
                self.criterion[key].cuda(args.local_rank)
        else:
            self.criterion.cuda(self.args.local_rank)

        # optimizer
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # save config file
        self.config_save_path = os.path.join(self.checkpoint_dir, "args.json")

    def build_logger(self):
        log_file = os.path.join(self.checkpoint_dir, "train.log")
        logger = Logger(log_file, level=self.args.log_level).logger

        return logger

    def build_dataset(self):
        raise NotImplementedError

    def build_data_loader(self):
        Dataset, Collator = self.build_dataset()
        # build dataset instance for each dataset and combine them by ConcatDataset
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=False)
            datasets_list.append(subdataset)
        train_dataset = ConcatDataset(datasets_list)

        train_collate = Collator(self.cfg)
        # TODO: multi-GPU training
        if self.cfg.train.ddp:
            raise NotImplementedError("DDP is not supported yet.")

        # sampler will provide indices to batch_sampler, which will perform batching and yield batch indices
        batch_sampler = BatchSampler(
            cfg=self.cfg, concat_dataset=train_dataset, dataset_list=datasets_list
        )

        # use batch_sampler argument instead of (sampler, shuffle, drop_last, batch_size)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_collate,
            num_workers=self.args.num_workers,
            batch_sampler=batch_sampler,
            pin_memory=False,
        )
        if not self.cfg.train.ddp or self.args.local_rank == 0:
            datasets_list = []
            for dataset in self.cfg.dataset:
                subdataset = Dataset(self.cfg, dataset, is_valid=True)
                datasets_list.append(subdataset)
            valid_dataset = ConcatDataset(datasets_list)
            valid_collate = Collator(self.cfg)
            batch_sampler = BatchSampler(
                cfg=self.cfg, concat_dataset=valid_dataset, dataset_list=datasets_list
            )
            valid_loader = DataLoader(
                valid_dataset,
                collate_fn=valid_collate,
                num_workers=1,
                batch_sampler=batch_sampler,
            )
        else:
            raise NotImplementedError("DDP is not supported yet.")
            # valid_loader = None
        data_loader = {"train": train_loader, "valid": valid_loader}
        return data_loader

    def build_singers_lut(self):
        # combine singers
        if not os.path.exists(os.path.join(self.log_dir, self.cfg.preprocess.spk2id)):
            singers = collections.OrderedDict()
        else:
            with open(
                os.path.join(self.log_dir, self.cfg.preprocess.spk2id), "r"
            ) as singer_file:
                singers = json.load(singer_file)
        singer_count = len(singers)
        for dataset in self.cfg.dataset:
            singer_lut_path = os.path.join(
                self.cfg.preprocess.processed_dir, dataset, self.cfg.preprocess.spk2id
            )
            with open(singer_lut_path, "r") as singer_lut_path:
                singer_lut = json.load(singer_lut_path)
            for singer in singer_lut.keys():
                if singer not in singers:
                    singers[singer] = singer_count
                    singer_count += 1
        with open(
            os.path.join(self.log_dir, self.cfg.preprocess.spk2id), "w"
        ) as singer_file:
            json.dump(singers, singer_file, indent=4, ensure_ascii=False)
        print(
            "singers have been dumped to {}".format(
                os.path.join(self.log_dir, self.cfg.preprocess.spk2id)
            )
        )
        return singers

    def build_model(self):
        raise NotImplementedError()

    def build_optimizer(self):
        raise NotImplementedError

    def build_scheduler(self):
        raise NotImplementedError()

    def build_criterion(self):
        raise NotImplementedError

    def get_state_dict(self):
        raise NotImplementedError

    def save_config_file(self):
        save_config(self.config_save_path, self.cfg)

    # TODO, save without module.
    def save_checkpoint(self, state_dict, saved_model_path):
        torch.save(state_dict, saved_model_path)

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint")
        assert os.path.exists(checkpoint_path)
        checkpoint_filename = open(checkpoint_path).readlines()[-1].strip()
        model_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        assert os.path.exists(model_path)
        if not self.cfg.train.ddp or self.args.local_rank == 0:
            self.logger.info(f"Re(store) from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        return checkpoint

    def load_model(self, checkpoint):
        raise NotImplementedError

    def restore(self):
        checkpoint = self.load_checkpoint()
        self.load_model(checkpoint)

    def train_step(self, data):
        raise NotImplementedError(
            f"Need to implement function {sys._getframe().f_code.co_name} in "
            f"your sub-class of {self.__class__.__name__}. "
        )

    @torch.no_grad()
    def eval_step(self):
        raise NotImplementedError(
            f"Need to implement function {sys._getframe().f_code.co_name} in "
            f"your sub-class of {self.__class__.__name__}. "
        )

    def write_summary(self, losses, stats):
        raise NotImplementedError(
            f"Need to implement function {sys._getframe().f_code.co_name} in "
            f"your sub-class of {self.__class__.__name__}. "
        )

    def write_valid_summary(self, losses, stats):
        raise NotImplementedError(
            f"Need to implement function {sys._getframe().f_code.co_name} in "
            f"your sub-class of {self.__class__.__name__}. "
        )

    def echo_log(self, losses, mode="Training"):
        message = [
            "{} - Epoch {} Step {}: [{:.3f} s/step]".format(
                mode, self.epoch + 1, self.step, self.time_window.average
            )
        ]

        for key in sorted(losses.keys()):
            if isinstance(losses[key], dict):
                for k, v in losses[key].items():
                    message.append(
                        str(k).split("/")[-1] + "=" + str(round(float(v), 5))
                    )
            else:
                message.append(
                    str(key).split("/")[-1] + "=" + str(round(float(losses[key]), 5))
                )
        self.logger.info(", ".join(message))

    def eval_epoch(self):
        self.logger.info("Validation...")
        valid_losses = {}
        for i, batch_data in enumerate(self.data_loader["valid"]):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.cuda()
            valid_loss, valid_stats, total_valid_loss = self.eval_step(batch_data, i)
            for key in valid_loss:
                if key not in valid_losses:
                    valid_losses[key] = 0
                valid_losses[key] += valid_loss[key]

        # Add mel and audio to the Tensorboard
        # Average loss
        for key in valid_losses:
            valid_losses[key] /= i + 1
        self.echo_log(valid_losses, "Valid")
        return valid_losses, valid_stats

    def train_epoch(self):
        for i, batch_data in enumerate(self.data_loader["train"]):
            start_time = time.time()
            # Put the data to cuda device
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.cuda(self.args.local_rank)

            # Training step
            train_losses, train_stats, total_loss = self.train_step(batch_data)
            self.time_window.append(time.time() - start_time)

            if self.args.local_rank == 0 or not self.cfg.train.ddp:
                if self.step % self.args.stdout_interval == 0:
                    self.echo_log(train_losses, "Training")

                if self.step % self.cfg.train.save_summary_steps == 0:
                    self.logger.info(f"Save summary as step {self.step}")
                    self.write_summary(train_losses, train_stats)

                if (
                    self.step % self.cfg.train.save_checkpoints_steps == 0
                    and self.step != 0
                ):
                    saved_model_name = "step-{:07d}_loss-{:.4f}.pt".format(
                        self.step, total_loss
                    )
                    saved_model_path = os.path.join(
                        self.checkpoint_dir, saved_model_name
                    )
                    saved_state_dict = self.get_state_dict()
                    self.save_checkpoint(saved_state_dict, saved_model_path)
                    self.save_config_file()
                    # keep max n models
                    remove_older_ckpt(
                        saved_model_name,
                        self.checkpoint_dir,
                        max_to_keep=self.cfg.train.keep_checkpoint_max,
                    )

                if self.step != 0 and self.step % self.cfg.train.valid_interval == 0:
                    if isinstance(self.model, dict):
                        for key in self.model.keys():
                            self.model[key].eval()
                    else:
                        self.model.eval()
                    # Evaluate one epoch and get average loss
                    valid_losses, valid_stats = self.eval_epoch()
                    if isinstance(self.model, dict):
                        for key in self.model.keys():
                            self.model[key].train()
                    else:
                        self.model.train()
                    # Write validation losses to summary.
                    self.write_valid_summary(valid_losses, valid_stats)
            self.step += 1

    def train(self):
        for epoch in range(max(0, self.epoch), self.max_epochs):
            self.train_epoch()
            self.epoch += 1
            if self.step > self.max_steps:
                self.logger.info("Training finished!")
                break
