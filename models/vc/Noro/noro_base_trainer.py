# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import torch
import time
from pathlib import Path
import torch
import accelerate
from accelerate.logging import get_logger
from models.base.new_trainer import BaseTrainer


class Noro_base_Trainer(BaseTrainer):

    def __init__(self, args=None, cfg=None):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        # init with accelerate
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        with self.accelerator.main_process_first():
            self.logger = get_logger(args.exp_name, log_level="INFO")

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
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

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

        # create criterion
        with self.accelerator.main_process_first():
            self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            self.logger.info(f"Building criterion done in {(end - start) / 1e6:.2f}ms")

        # Resume or Finetune
        with self.accelerator.main_process_first():
            self._check_resume()

        # accelerate prepare
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        self._accelerator_prepare()
        end = time.monotonic_ns()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")
        self.device = self.accelerator.device

        if cfg.preprocess.use_spkid and cfg.train.multi_speaker_training:
            self.speakers = self._build_speaker_lut()
            self.utt2spk_dict = self._build_utt2spk_dict()

        self.task_type = "VC"
        self.logger.info("Task type: {}".format(self.task_type))

    def _check_resume(self):
        # if args.resume:
        if self.args.resume:
            self.logger.info("Resuming from checkpoint...")
            self.ckpt_path = self._load_model(
                self.checkpoint_dir, self.args.checkpoint_path, self.args.resume_type
            )
            self.checkpoints_path = json.load(
                open(os.path.join(self.ckpt_path, "ckpts.json"), "r")
            )

    def _load_model(self, checkpoint_dir, checkpoint_path=None, resume_type="resume"):
        """Load model from checkpoint. If a folder is given, it will
        load the latest checkpoint in checkpoint_dir. If a path is given
        it will load the checkpoint specified by checkpoint_path.
        **Only use this method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None or checkpoint_path == "":
            ls = [str(i) for i in Path(checkpoint_dir).glob("*")]
            checkpoint_path = max(
                ls, key=lambda x: int(x.split("_")[-2].split("-")[-1])
            )

        if self.accelerator.is_main_process:
            self.logger.info("Load model from {}".format(checkpoint_path))
        print("Load model from {}".format(checkpoint_path))

        if resume_type == "resume":
            self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1])
            self.step = int(checkpoint_path.split("_")[-2].split("-")[-1])
            if isinstance(self.model, dict):
                for idx, sub_model in enumerate(self.model.keys()):
                    try:
                        if idx == 0:
                            ckpt_name = "pytorch_model.bin"
                        else:
                            ckpt_name = "pytorch_model_{}.bin".format(idx)

                        self.model[sub_model].load_state_dict(
                            torch.load(os.path.join(checkpoint_path, ckpt_name))
                        )
                    except:
                        if idx == 0:
                            ckpt_name = "model.safetensors"
                        else:
                            ckpt_name = "model_{}.safetensors".format(idx)

                        accelerate.load_checkpoint_and_dispatch(
                            self.accelerator.unwrap_model(self.model[sub_model]),
                            os.path.join(checkpoint_path, ckpt_name),
                        )

                self.model[sub_model].cuda(self.accelerator.device)
            else:
                try:
                    self.model.load_state_dict(
                        torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
                    )
                    if self.accelerator.is_main_process:
                        self.logger.info("Loaded 'pytorch_model.bin' for resume")
                except:
                    accelerate.load_checkpoint_and_dispatch(
                        self.accelerator.unwrap_model(self.model),
                        os.path.join(checkpoint_path, "model.safetensors"),
                    )
                    if self.accelerator.is_main_process:
                        self.logger.info("Loaded 'model.safetensors' for resume")
                self.model.cuda(self.accelerator.device)
            if self.accelerator.is_main_process:
                self.logger.info("Load model weights SUCCESS!")
        elif resume_type == "finetune":
            if isinstance(self.model, dict):
                for idx, sub_model in enumerate(self.model.keys()):
                    try:
                        if idx == 0:
                            ckpt_name = "pytorch_model.bin"
                        else:
                            ckpt_name = "pytorch_model_{}.bin".format(idx)

                        self.model[sub_model].load_state_dict(
                            torch.load(os.path.join(checkpoint_path, ckpt_name))
                        )
                    except:
                        if idx == 0:
                            ckpt_name = "model.safetensors"
                        else:
                            ckpt_name = "model_{}.safetensors".format(idx)

                        accelerate.load_checkpoint_and_dispatch(
                            self.accelerator.unwrap_model(self.model[sub_model]),
                            os.path.join(checkpoint_path, ckpt_name),
                        )

                self.model[sub_model].cuda(self.accelerator.device)
            else:
                try:
                    self.model.load_state_dict(
                        torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
                    )
                    if self.accelerator.is_main_process:
                        self.logger.info("Loaded 'pytorch_model.bin' for finetune")
                except:
                    accelerate.load_checkpoint_and_dispatch(
                        self.accelerator.unwrap_model(self.model),
                        os.path.join(checkpoint_path, "model.safetensors"),
                    )
                    if self.accelerator.is_main_process:
                        self.logger.info("Loaded 'model.safetensors' for finetune")
                self.model.cuda(self.accelerator.device)
            if self.accelerator.is_main_process:
                self.logger.info("Load model weights for finetune SUCCESS!")
        else:
            raise ValueError("Unsupported resume type: {}".format(resume_type))
        return checkpoint_path

    def _check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            self.logger.fatal("Invalid gradient_accumulation_step value!")
            self.logger.error(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
