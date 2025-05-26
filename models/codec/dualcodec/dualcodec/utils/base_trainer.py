# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import shutil
import time
from abc import abstractmethod
from pathlib import Path
import math
import accelerate
import numpy as np
import torch
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm
import psutil
import torch.distributed as dist


def get_memory_usage():
    """
    Function to get memory usage in GB
    """
    # Process memory usage
    process = psutil.Process()
    mem_info = process.memory_info()
    proc_mem_rss = mem_info.rss / (1024**3)  # Convert bytes to GB

    # System memory usage
    mem = psutil.virtual_memory()
    total_mem = mem.total / (1024**3)  # Convert bytes to GB
    available_mem = mem.available / (1024**3)  # Convert bytes to GB
    used_mem = mem.used / (1024**3)  # Convert bytes to GB
    free_mem = mem.free / (1024**3)  # Convert bytes to GB

    return {
        "CPU/Total (GB)": total_mem,
        "CPU/Available (GB)": available_mem,
        "CPU/Used (GB)": used_mem,
        "CPU/Free (GB)": free_mem,
        "CPU/Process RSS (GB)": proc_mem_rss,
    }


def get_gpu_memory_usage():
    """get gpu memory"""
    gpu_memory_usage = {}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(1):
            gpu_memory_usage[f"GPU/GPU_{i}/Allocated (GB)"] = (
                torch.cuda.memory_allocated(i) / (1024**3)
            )
            gpu_memory_usage[f"GPU/GPU_{i}/Reserved (GB)"] = torch.cuda.memory_reserved(
                i
            ) / (1024**3)
    return gpu_memory_usage


class MainProcessLogger:
    """Logger"""

    def __init__(self, is_main_process=True, name=None, **kwargs):
        """
        Args:
        is_main_process (bool, optional): Whether the process is main process or not. Defaults to True.
        name (str, optional): The name of the logger. Defaults to None.
        kwargs (dict, optional): Other keyword arguments for logging.Default to {}.
        """
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
        """
            Initializes a new instance of `TrainingContext`.

        Args:
            args (Optional[argparse.Namespace], optional): Arguments dictionary. Defaults to None.
            cfg (Optional[Dict], optional): Configuration dictionary. Defaults to None.

        Raises:
            ValueError: If the resumed checkpoint path doesn't exist when resuming.
        """
        super().__init__()

        self.args = args
        self.cfg = cfg
        self.raise_oom = False
        if hasattr(self.cfg, "raise_oom"):
            self.raise_oom = self.cfg.raise_oom

        cfg.exp_name = args.exp_name

        # init with accelerate
        print("initializing accelerator...")
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        self.is_dataset_ready = False  # NOTE: requires calling `_build_dataloader`
        # after this __init__

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
        self.current_loss = 0.0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        self.max_steps = (
            self.cfg.train.max_steps
            if hasattr(self.cfg.train, "max_steps")
            else float("inf")
        )
        if self.max_steps <= 0 or self.max_steps is None:
            self.max_steps = float("inf")
        print("max steps:", self.max_steps)
        self.logger.info(
            "Max epoch: {}".format(
                self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
            )
        )

        # Check values
        if self.accelerator.is_main_process:
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
        self._set_random_seed(args.seed)

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
                        checkpoint_dir=self.checkpoint_dir, resume_type="resume"
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
        self.config_save_path = os.path.join(self.exp_dir, "args.yaml")
        # if self.accelerator.is_main_process:
        #     print(self.cfg)

    def _accelerator_prepare(self):
        (
            self.model,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
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

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            keep_last = self.keep_last[0]
            all_ckpts = os.listdir(self.checkpoint_dir)
            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            all_ckpts = list(all_ckpts)
            if len(all_ckpts) > keep_last:
                # only keep `keep_last` folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                all_ckpts = sorted(
                    all_ckpts, key=lambda x: int(x.split("_")[1].split("-")[1])
                )
                for ckpt in all_ckpts[:-keep_last]:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))
            checkpoint_filename = f"epoch-{self.epoch:04d}_step-{self.step:07d}_loss-{self.current_loss:.6f}-{self.args.exp_name}"
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            self.logger.info("Saving state to {}...".format(path))
            self.accelerator.save_state(path)
            self.logger.info("Finished saving state.")

    @abstractmethod
    def _save_auxiliary_states(self):
        r"""To save some auxiliary states when saving model's ckpt"""
        pass

    ### Abstract methods end ###

    ### THIS IS MAIN ENTRY ###
    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        assert (
            self.is_dataset_ready
        ), "make sure to call _build_dataloader to prepare the data!"
        # Wait everyone to prepare before we move on
        print(
            f"Process {self.accelerator.process_index} beginning to wait for everyone before training..."
        )
        self.accelerator.wait_for_everyone()
        print(f"Process {self.accelerator.process_index} finished waiting!")
        # dump config file
        self.model.train()
        self.optimizer.zero_grad()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            self._train_epoch()
            print(
                f"Process {self.accelerator.process_index} is finishing epoch, max_epoch is {self.max_epoch}"
            )
            # self._valid_epoch()

            # self.accelerator.wait_for_everyone()

            self.epoch += 1

        self.accelerator.end_training()

    def get_lr(self, it):
        """
        get cosine lr with warmup
        """
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

    def log(self, *args, **kwargs):
        """
        log every 200 steps
        """
        try:
            if self.step % 200 == 0 and self.accelerator.is_main_process:
                self.accelerator.log(*args, **kwargs)
        except Exception as e:
            print(e)

    ### Following are methods that can be used directly in child classes ###
    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.train()
        ema_loss = None

        # profiler
        start_this_step_time = time.time()
        finish_last_step_time = time.time()

        num_finished_samples = 0  # number of sample passes
        trained_data_duration = 0

        data_iter = iter(self.train_dataloader)
        data_idx = 0

        pbar = tqdm(
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
            mininterval=15,
        )
        while True:
            try:
                batch = next(data_iter)
                pbar.update(1)
                data_idx += 1
            except StopIteration:
                print("End of DataLoader reached for the epoch.")
                break
            except Exception as e:
                print(e)
                continue
            start_this_step_time = time.time()
            # print(f'load batch took: {start_this_step_time - finish_last_step_time:.6f}s')
            trained_data_duration += (
                batch["duration"] / 3600
            ) * self.accelerator.num_processes
            stats = {
                "Time/load batch": start_this_step_time - finish_last_step_time,
                "Data/duration": trained_data_duration,
                "Time/load audio": batch["load_audio_time"],
            }
            self.log(stats, step=self.step)
            if stats["Time/load batch"] > 0.6:
                print(
                    f'{self.accelerator.process_index} load batch time too long, took {stats["Time/load batch"]}s'
                )

            # update learning rate
            lr = self.get_lr(self.step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            # Do training step and BP
            if data_idx == 1:
                print(f"Process {self.accelerator.process_index} is before first step")
            with self.accelerator.accumulate(self.model):
                try:
                    loss = self._train_step(batch)
                    if data_idx == 1:
                        print(
                            f"Process {self.accelerator.process_index} completed first step"
                        )
                except RuntimeError as e:
                    if "out of memory" in str(e) and not self.raise_oom:
                        print("| WARNING: ran out of memory, passing batch")
                        self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                forward_this_step_time = time.time()
                if loss is not None:
                    if type(loss) is tuple:
                        stats = loss[1]
                        self.log(stats, step=self.step)
                        if "Data/Batch Size" in stats:
                            num_finished_samples += int(stats["Data/Batch Size"])
                        loss = loss[0]
                # self.accelerator.wait_for_everyone()
                sync_time = time.time()
                if loss is not None:
                    self.current_loss = loss.item()
                    ema_loss = (
                        0.99 * ema_loss + 0.01 * self.current_loss
                        if ema_loss is not None
                        else self.current_loss
                    )
                    if torch.isnan(loss):
                        print(f"Nan loss encountered in step {self.step}!")
                        self.optimizer.zero_grad()
                        continue
                    if hasattr(self.cfg, "ignore_outlier_loss_factor"):
                        if (
                            float(self.cfg.ignore_outlier_loss_factor) * ema_loss < loss
                            and self.step > 5000
                        ):
                            print(f"ignoring outlier loss with value {loss.item()}")
                            continue
                    self.accelerator.backward(loss)
                    backward_time = time.time()
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    optimizer_step_time = time.time()
                else:
                    loss = 0.0
                    ema_loss = 0.0
                    backward_time = sync_time
                    optimizer_step_time = sync_time
            self.batch_count += 1

            # if self.accelerator.is_main_process:
            #     print(self.current_loss)

            if self.accelerator.sync_gradients:
                if (
                    self.step % self.cfg.train.save_checkpoint_stride[0] == 0
                    or self.step == self.max_steps
                ):
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        try:
                            self.save_checkpoint()
                        except Exception as e:
                            print(e)
                            self.logger.info("Failed to save checkpoint, resuming...")
                    # if self.step == self.max_steps:
                    #     exit(0)
                if self.accelerator.is_main_process:
                    if self.step % 200 == 0:
                        self.logger.info(f"Running Avg Loss: {ema_loss:.5f}")
                        # self.log(get_memory_usage(), step=self.step)
                        self.log(get_gpu_memory_usage(), step=self.step)
                        try:
                            self.log(
                                {
                                    "Step/Epoch": batch["epoch"]
                                    + num_finished_samples / len(self.train_dataloader)
                                },
                                step=self.step,
                            )
                        except Exception as e:
                            # print(e)
                            pass
                if self.step % 200 == 0:
                    self.log(
                        {
                            "Step/Train Loss": loss,
                            "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self.step,
                    )
                self.step += 1

            finish_last_step_time = time.time()
            stats = {
                "Time/forward_backward": finish_last_step_time - start_this_step_time,
                "Time/process_sync": sync_time - forward_this_step_time,
                "Time/backward": backward_time - sync_time,
                "Time/optimizer_step": optimizer_step_time - backward_time,
                "Time/forward": forward_this_step_time - start_this_step_time,
            }
            if self.step % 200 == 0:
                self.log(stats, step=self.step)

            # emptying the CUDA cache after the first step can
            # reduce the chance of OOM

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
            mininterval=15,
        ):
            batch_loss = self._valid_step(batch)
            epoch_sum_loss += batch_loss.item()

        return epoch_sum_loss / len(self.valid_dataloader)

    @abstractmethod
    def _train_step(self, batch):
        r"""Training forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_train_epoch`` for usage.
        """
        pass

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
            try:
                self.accelerator.load_state(input_dir=checkpoint_path)
            except Exception as e:
                print(e)
            # set epoch and step
            from pathlib import Path

            self.epoch = int(Path(checkpoint_path).name.split("_")[0].split("-")[-1])
            if hasattr(self.args, "reset_steps") and self.args.reset_steps:
                self.step = 0
            else:
                self.step = (
                    int(Path(checkpoint_path).name.split("_")[1].split("-")[-1]) + 1
                )

        elif resume_type == "finetune":
            # Load only the model weights
            accelerate.load_checkpoint_and_dispatch(
                self.accelerator.unwrap_model(self.model),
                os.path.join(checkpoint_path, "model.safetensors"),
            )
            self.logger.info("Load model weights for finetune...")

        else:
            raise ValueError("Resume_type must be `resume` or `finetune`.")

        return checkpoint_path

    def _build_dataloader(self, dataloader):
        # setup data_loader
        self.logger.info("Building dataset...")
        start = time.monotonic_ns()
        self.train_dataloader, self.valid_dataloader = dataloader, None
        end = time.monotonic_ns()
        self.is_dataset_ready = True
        self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

    @staticmethod
    def _set_random_seed(seed):
        r"""Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def _build_optimizer(self):
        r"""Build optimizer for model."""
        return self.cfg.train.optimizer(params=self.model.parameters())

    def _build_scheduler(self):
        r"""Build scheduler for optimizer."""
        raise NotImplementedError

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
        from datetime import timedelta

        kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.cfg.train.find_unused_parameters
        )
        process_group_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=3600 * 12)
        )
        mixed_precision = "fp16"
        if (
            hasattr(self.cfg.train, "disable_mixed_precision")
            and self.cfg.train.disable_mixed_precision
        ):
            print("using fp32...")
            mixed_precision = "no"
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            kwargs_handlers=[kwargs, process_group_handler],
            mixed_precision=mixed_precision,
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(project_name=self.args.exp_name)

    @staticmethod
    def __count_parameters(model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    @torch.inference_mode()
    def test_loop(self):
        """
        Tests the loop functionality of the model.
        """
        pass

    ### Private methods end ###
