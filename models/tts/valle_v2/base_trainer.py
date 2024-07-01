# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import shutil
import time
from abc import abstractmethod
from pathlib import Path
import math
import accelerate
import json5
import numpy as np
import torch
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from models.base.base_sampler import build_samplers
from optimizer.optimizers import NoamLR


class MainProcessLogger:
    def __init__(self, is_main_process=True, name=None, **kwargs):
        import logging

        if name is None:
            logger = logging.getLogger(__name__)
        else:
            logger = logging.getLogger(name)
        self.logger = logger
        self.is_main_process = is_main_process

    def info(self, msg):
        if self.is_main_process:
            print(msg)
            # self.logger.info(msg)

    def debug(self, msg):
        if self.is_main_process:
            print(msg)
            # self.logger.debug(msg)

    def warning(self, msg):
        if self.is_main_process:
            print(msg)
            # self.logger.warning(msg)


class BaseTrainer(object):
    r"""The base trainer for all tasks. Any trainer should inherit from this class."""

    def __init__(self, args=None, cfg=None):
        super().__init__()

        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        # init with accelerate
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Use accelerate logger for distributed training
        with self.accelerator.main_process_first():
            self.logger = MainProcessLogger(
                self.accelerator.is_main_process,
                name=args.exp_name,
                log_level=args.log_level,
            )

        # Log some info
        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New training process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")
        self.logger.debug(f"Using {args.log_level.upper()} logging level.")
        self.logger.info(f"Experiment name: {args.exp_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        self.logger.info(
            "Max epoch: {}".format(
                self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
            )
        )

        # Check values
        if self.accelerator.is_main_process:
            self.__check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(args.seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {args.seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # setup model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            self.logger.debug(self.model)
            self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
            self.logger.info(
                f"Model parameters: {self.__count_parameters(self.model)/1e6:.2f}M"
            )
        # optimizer & scheduler
        with self.accelerator.main_process_first():
            self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            self.logger.info(
                f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
            )

        # accelerate prepare
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        self._accelerator_prepare()
        end = time.monotonic_ns()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # create criterion
        with self.accelerator.main_process_first():
            self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            self.logger.info(f"Building criterion done in {(end - start) / 1e6:.2f}ms")

        # Resume or Finetune
        with self.accelerator.main_process_first():
            if args.resume:
                if args.resume_from_ckpt_path == "":
                    ## Automatically resume according to the current exprimental name
                    self.logger.info(
                        "Automatically resuming from latest checkpoint in {}...".format(
                            self.checkpoint_dir
                        )
                    )
                    start = time.monotonic_ns()
                    ckpt_path = self._load_model(
                        checkpoint_dir=self.checkpoint_dir, resume_type=args.resume_type
                    )
                    end = time.monotonic_ns()
                    self.logger.info(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )
                else:
                    ## Resume from the given checkpoint path
                    if not os.path.exists(args.resume_from_ckpt_path):
                        raise ValueError(
                            "[Error] The resumed checkpoint path {} don't exist.".format(
                                args.resume_from_ckpt_path
                            )
                        )
                    self.logger.info(
                        "Resuming from {}...".format(args.resume_from_ckpt_path)
                    )
                    start = time.monotonic_ns()
                    ckpt_path = self._load_model(
                        checkpoint_path=args.resume_from_ckpt_path,
                        resume_type=args.resume_type,
                    )
                    end = time.monotonic_ns()
                    self.logger.info(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

    def _accelerator_prepare(self):
        (
            self.train_dataloader,
            self.valid_dataloader,
            self.model,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.valid_dataloader,
            self.model,
            self.optimizer,
            self.scheduler,
        )

    ### Following are abstract methods that should be implemented in child classes ###
    @abstractmethod
    def _build_dataset(self):
        r"""Build dataset for model training/validating/evaluating."""
        pass

    @staticmethod
    @abstractmethod
    def _build_criterion():
        r"""Build criterion function for model loss calculation."""
        pass

    @abstractmethod
    def _build_model(self):
        r"""Build model for training/validating/evaluating."""
        pass

    @abstractmethod
    def _forward_step(self, batch):
        r"""One forward step of the neural network. This abstract method is trying to
        unify ``_train_step`` and ``_valid_step`` and avoid redundant implementation.
        However, for special case that using different forward step pattern for
        training and validating, you could just override this method with ``pass`` and
        implement ``_train_step`` and ``_valid_step`` separately.
        """
        pass

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            keep_last = self.keep_last[0]
            # 读取self.checkpoint_dir所有的folder
            all_ckpts = os.listdir(self.checkpoint_dir)
            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            all_ckpts = list(all_ckpts)
            if len(all_ckpts) > keep_last:
                # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                all_ckpts = sorted(
                    all_ckpts, key=lambda x: int(x.split("_")[1].split("-")[1])
                )
                for ckpt in all_ckpts[:-keep_last]:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))
            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            self.logger.info("Saving state to {}...".format(path))
            self.accelerator.save_state(path)
            self.logger.info("Finished saving state.")

    @abstractmethod
    def _save_auxiliary_states(self):
        r"""To save some auxiliary states when saving model's ckpt"""
        pass

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

    ### Abstract methods end ###

    ### THIS IS MAIN ENTRY ###
    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        if self.accelerator.is_main_process:
            self.__dump_cfg(self.config_save_path)
        self.model.train()
        self.optimizer.zero_grad()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            ### TODO: change the return values of _train_epoch() to a loss dict, or (total_loss, loss_dict)
            ### It's inconvenient for the model with multiple losses
            # Do training & validating epoch
            train_loss = self._train_epoch()
            self.logger.info("  |- Train/Loss: {:.6f}".format(train_loss))
            valid_loss = self._valid_epoch()
            self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_loss))
            self.accelerator.log(
                {"Epoch/Train Loss": train_loss, "Epoch/Valid Loss": valid_loss},
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_loss
                    ),
                )
            )
            self._save_auxiliary_states()

        self.accelerator.end_training()

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.cfg.train.scheduler.warmup_steps:
            return self.cfg.train.adamw.lr * it / self.cfg.train.scheduler.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.cfg.train.scheduler.total_steps:
            return self.cfg.train.scheduler.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.cfg.train.scheduler.warmup_steps) / (
            self.cfg.train.scheduler.total_steps - self.cfg.train.scheduler.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.cfg.train.scheduler.min_lr + coeff * (
            self.cfg.train.adamw.lr - self.cfg.train.scheduler.min_lr
        )

    ### Following are methods that can be used directly in child classes ###
    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.train()
        epoch_sum_loss: float = 0.0
        ema_loss = None

        # profiler
        start_this_step_time = time.time()
        finish_last_step_time = time.time()

        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            assert batch is not None

            # start_this_step_time = time.time()
            # print(f'load batch took: {start_this_step_time - finish_last_step_time:.6f}s')

            # update learning rate
            lr = self.get_lr(self.step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                loss = self._train_step(batch)
                self.current_loss = loss.item()
                ema_loss = (
                    0.99 * ema_loss + 0.01 * self.current_loss
                    if ema_loss is not None
                    else self.current_loss
                )
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.batch_count += 1

            # if self.accelerator.is_main_process:
            #     print(self.current_loss)

            if self.accelerator.sync_gradients:
                if self.step % self.cfg.train.save_checkpoint_stride[0] == 0:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        try:
                            self.save_checkpoint()
                        except:
                            self.logger.info("Failed to save checkpoint, resuming...")
                if self.accelerator.is_main_process:
                    if self.step % 100 == 0:
                        self.logger.info(f"EMA Loss: {ema_loss:.6f}")
                self.accelerator.log(
                    {
                        "Step/Train Loss": loss,
                        "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.step,
                )
                epoch_sum_loss += loss
                self.step += 1

            # finish_last_step_time = time.time()
            # print(f'load took: {finish_last_step_time - start_this_step_time:.6f}s')
        return (
            epoch_sum_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.eval()
        epoch_sum_loss = 0.0
        for batch in tqdm(
            self.valid_dataloader,
            desc=f"Validating Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            batch_loss = self._valid_step(batch)
            epoch_sum_loss += batch_loss.item()

        return epoch_sum_loss / len(self.valid_dataloader)

    def _train_step(self, batch):
        r"""Training forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_train_epoch`` for usage.
        """
        return self._forward_step(batch)

    @torch.inference_mode()
    def _valid_step(self, batch):
        r"""Testing forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_test_epoch`` for usage.
        """
        return self._forward_step(batch)

    def _load_model(
        self,
        checkpoint_dir: str = None,
        checkpoint_path: str = None,
        resume_type: str = "",
    ):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            try:
                all_ckpts = os.listdir(checkpoint_dir)
                all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
                ls = list(all_ckpts)
                ls = [os.path.join(checkpoint_dir, i) for i in ls]
                ls.sort(
                    key=lambda x: int(x.split("_")[-2].split("-")[-1]), reverse=True
                )
                checkpoint_path = ls[0]
                self.logger.info("Resume from {}".format(checkpoint_path))
            except Exception as e:
                print(
                    "Failed to load checkpoint from {}, starting FROM SCRATCH...".format(
                        checkpoint_dir
                    )
                )
                return None

        if resume_type in ["resume", ""]:
            # Load all the things, including model weights, optimizer, scheduler, and random states.
            self.accelerator.load_state(input_dir=checkpoint_path)

            # set epoch and step
            self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1]) + 1
            self.step = int(checkpoint_path.split("_")[-2].split("-")[-1]) + 1

        elif resume_type == "finetune":
            # Load only the model weights
            accelerate.load_checkpoint_and_dispatch(
                self.accelerator.unwrap_model(self.model),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
            )
            self.logger.info("Load model weights for finetune...")

        else:
            raise ValueError("Resume_type must be `resume` or `finetune`.")

        return checkpoint_path

    # TODO: LEGACY CODE
    def _build_dataloader(self):
        Dataset, Collator = self._build_dataset()

        # build dataset instance for each dataset and combine them by ConcatDataset
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=False)
            datasets_list.append(subdataset)
        train_dataset = ConcatDataset(datasets_list)
        train_collate = Collator(self.cfg)
        _, batch_sampler = build_samplers(train_dataset, self.cfg, self.logger, "train")
        self.logger.debug(f"train batch_sampler: {list(batch_sampler)}")
        self.logger.debug(f"length: {train_dataset.cumulative_sizes}")
        # TODO: use config instead of (sampler, shuffle, drop_last, batch_size)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_collate,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.train.dataloader.num_worker,
            pin_memory=self.cfg.train.dataloader.pin_memory,
        )

        # Build valid dataloader
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=True)
            datasets_list.append(subdataset)
        valid_dataset = ConcatDataset(datasets_list)
        valid_collate = Collator(self.cfg)
        _, batch_sampler = build_samplers(valid_dataset, self.cfg, self.logger, "valid")
        self.logger.debug(f"valid batch_sampler: {list(batch_sampler)}")
        self.logger.debug(f"length: {valid_dataset.cumulative_sizes}")
        valid_loader = DataLoader(
            valid_dataset,
            collate_fn=valid_collate,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.train.dataloader.num_worker,
            pin_memory=self.cfg.train.dataloader.pin_memory,
        )
        return train_loader, valid_loader

    @staticmethod
    def _set_random_seed(seed):
        r"""Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def _check_nan(self, loss, y_pred, y_gt):
        if torch.any(torch.isnan(loss)):
            self.logger.fatal("Fatal Error: Training is down since loss has Nan!")
            self.logger.error("loss = {:.6f}".format(loss.item()), in_order=True)
            if torch.any(torch.isnan(y_pred)):
                self.logger.error(
                    f"y_pred has Nan: {torch.any(torch.isnan(y_pred))}", in_order=True
                )
            else:
                self.logger.debug(
                    f"y_pred has Nan: {torch.any(torch.isnan(y_pred))}", in_order=True
                )
            if torch.any(torch.isnan(y_gt)):
                self.logger.error(
                    f"y_gt has Nan: {torch.any(torch.isnan(y_gt))}", in_order=True
                )
            else:
                self.logger.debug(
                    f"y_gt has nan: {torch.any(torch.isnan(y_gt))}", in_order=True
                )
            if torch.any(torch.isnan(y_pred)):
                self.logger.error(f"y_pred: {y_pred}", in_order=True)
            else:
                self.logger.debug(f"y_pred: {y_pred}", in_order=True)
            if torch.any(torch.isnan(y_gt)):
                self.logger.error(f"y_gt: {y_gt}", in_order=True)
            else:
                self.logger.debug(f"y_gt: {y_gt}", in_order=True)

            # TODO: still OK to save tracking?
            self.accelerator.end_training()
            raise RuntimeError("Loss has Nan! See log for more info.")

    ### Protected methods end ###

    ## Following are private methods ##
    ## !!! These are inconvenient for GAN-based model training. It'd be better to move these to svc_trainer.py if needed.
    def _build_optimizer(self):
        r"""Build optimizer for model."""
        # Make case-insensitive matching
        if self.cfg.train.optimizer.lower() == "adadelta":
            optimizer = torch.optim.Adadelta(
                self.model.parameters(), **self.cfg.train.adadelta
            )
            self.logger.info("Using Adadelta optimizer.")
        elif self.cfg.train.optimizer.lower() == "adagrad":
            optimizer = torch.optim.Adagrad(
                self.model.parameters(), **self.cfg.train.adagrad
            )
            self.logger.info("Using Adagrad optimizer.")
        elif self.cfg.train.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **self.cfg.train.adam)
            self.logger.info("Using Adam optimizer.")
        elif self.cfg.train.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), **self.cfg.train.adamw
            )
        elif self.cfg.train.optimizer.lower() == "sparseadam":
            optimizer = torch.optim.SparseAdam(
                self.model.parameters(), **self.cfg.train.sparseadam
            )
        elif self.cfg.train.optimizer.lower() == "adamax":
            optimizer = torch.optim.Adamax(
                self.model.parameters(), **self.cfg.train.adamax
            )
        elif self.cfg.train.optimizer.lower() == "asgd":
            optimizer = torch.optim.ASGD(self.model.parameters(), **self.cfg.train.asgd)
        elif self.cfg.train.optimizer.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.model.parameters(), **self.cfg.train.lbfgs
            )
        elif self.cfg.train.optimizer.lower() == "nadam":
            optimizer = torch.optim.NAdam(
                self.model.parameters(), **self.cfg.train.nadam
            )
        elif self.cfg.train.optimizer.lower() == "radam":
            optimizer = torch.optim.RAdam(
                self.model.parameters(), **self.cfg.train.radam
            )
        elif self.cfg.train.optimizer.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.model.parameters(), **self.cfg.train.rmsprop
            )
        elif self.cfg.train.optimizer.lower() == "rprop":
            optimizer = torch.optim.Rprop(
                self.model.parameters(), **self.cfg.train.rprop
            )
        elif self.cfg.train.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), **self.cfg.train.sgd)
        else:
            raise NotImplementedError(
                f"Optimizer {self.cfg.train.optimizer} not supported yet!"
            )
        return optimizer

    def _build_scheduler(self):
        r"""Build scheduler for optimizer."""
        # Make case-insensitive matching
        if self.cfg.train.scheduler.lower() == "lambdalr":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, **self.cfg.train.lambdalr
            )
        elif self.cfg.train.scheduler.lower() == "multiplicativelr":
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                self.optimizer, **self.cfg.train.multiplicativelr
            )
        elif self.cfg.train.scheduler.lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, **self.cfg.train.steplr
            )
        elif self.cfg.train.scheduler.lower() == "multisteplr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, **self.cfg.train.multisteplr
            )
        elif self.cfg.train.scheduler.lower() == "constantlr":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, **self.cfg.train.constantlr
            )
        elif self.cfg.train.scheduler.lower() == "linearlr":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, **self.cfg.train.linearlr
            )
        elif self.cfg.train.scheduler.lower() == "exponentiallr":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, **self.cfg.train.exponentiallr
            )
        elif self.cfg.train.scheduler.lower() == "polynomiallr":
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer, **self.cfg.train.polynomiallr
            )
        elif self.cfg.train.scheduler.lower() == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **self.cfg.train.cosineannealinglr
            )
        elif self.cfg.train.scheduler.lower() == "sequentiallr":
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, **self.cfg.train.sequentiallr
            )
        elif self.cfg.train.scheduler.lower() == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **self.cfg.train.reducelronplateau
            )
        elif self.cfg.train.scheduler.lower() == "cycliclr":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, **self.cfg.train.cycliclr
            )
        elif self.cfg.train.scheduler.lower() == "onecyclelr":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, **self.cfg.train.onecyclelr
            )
        elif self.cfg.train.scheduler.lower() == "cosineannearingwarmrestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, **self.cfg.train.cosineannearingwarmrestarts
            )
        elif self.cfg.train.scheduler.lower() == "noamlr":
            scheduler = NoamLR(self.optimizer, **self.cfg.train.lr_scheduler)
        else:
            raise NotImplementedError(
                f"Scheduler {self.cfg.train.scheduler} not supported yet!"
            )
        return scheduler

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        from accelerate import DistributedDataParallelKwargs

        kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.cfg.train.find_unused_parameters
        )

        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            kwargs_handlers=[kwargs],
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def __check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            self.logger.fatal("Invalid gradient_accumulation_step value!")
            self.logger.error(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
        # TODO: check other values

    @staticmethod
    def __count_parameters(model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def __dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )

    @torch.inference_mode()
    def test_loop(self):
        pass

    ### Private methods end ###
