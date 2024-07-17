# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import torch
import time
from pathlib import Path
import torch
from tqdm import tqdm
import re
import json5
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch.utils.data import ConcatDataset, DataLoader
from accelerate import DistributedDataParallelKwargs
from schedulers.scheduler import Eden
from models.base.base_sampler import build_samplers
from models.base.new_trainer import BaseTrainer


class TTSTrainer(BaseTrainer):
    r"""The base trainer for all TTS models. It inherits from BaseTrainer and implements
    ``build_criterion``, ``_build_dataset`` and ``_build_singer_lut`` methods. You can inherit from this
    class, and implement ``_build_model``, ``_forward_step``.
    """

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
            # start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            # self.logger.debug(
            #     f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            # )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # # save phone table to exp dir. Should be done before building model due to loading phone table in model
        # if cfg.preprocess.use_phone and cfg.preprocess.phone_extractor != "lexicon":
        #     self._save_phone_symbols_file_to_exp_path()

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

        # Only for TTS tasks
        self.task_type = "TTS"
        self.logger.info("Task type: {}".format(self.task_type))

    def _check_resume(self):
        # if args.resume:
        if self.args.resume or (
            self.cfg.model_type == "VALLE" and self.args.train_stage == 2
        ):
            if self.cfg.model_type == "VALLE" and self.args.train_stage == 2:
                self.args.resume_type = "finetune"

            self.logger.info("Resuming from checkpoint...")
            self.ckpt_path = self._load_model(
                self.checkpoint_dir, self.args.checkpoint_path, self.args.resume_type
            )
            self.checkpoints_path = json.load(
                open(os.path.join(self.ckpt_path, "ckpts.json"), "r")
            )

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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

    def _accelerator_prepare(self):
        (
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.valid_dataloader,
        )

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

    ### Following are methods only for TTS tasks ###
    def _build_dataset(self):
        pass

    def _build_criterion(self):
        pass

    def _build_model(self):
        pass

    def _build_dataloader(self):
        """Build dataloader which merges a series of datasets."""
        # Build dataset instance for each dataset and combine them by ConcatDataset
        Dataset, Collator = self._build_dataset()

        # Build train set
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=False)
            datasets_list.append(subdataset)
        train_dataset = ConcatDataset(datasets_list)
        train_collate = Collator(self.cfg)
        _, batch_sampler = build_samplers(train_dataset, self.cfg, self.logger, "train")
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_collate,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.train.dataloader.num_worker,
            pin_memory=self.cfg.train.dataloader.pin_memory,
        )

        # Build test set
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=True)
            datasets_list.append(subdataset)
        valid_dataset = ConcatDataset(datasets_list)
        valid_collate = Collator(self.cfg)
        _, batch_sampler = build_samplers(valid_dataset, self.cfg, self.logger, "valid")
        valid_loader = DataLoader(
            valid_dataset,
            collate_fn=valid_collate,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.train.dataloader.num_worker,
            pin_memory=self.cfg.train.dataloader.pin_memory,
        )
        return train_loader, valid_loader

    def _build_optimizer(self):
        pass

    def _build_scheduler(self):
        pass
    
    def _load_model(self, checkpoint_dir, checkpoint_path=None, resume_type="resume"):
        """Load model from checkpoint. If a folder is given, it will
        load the latest checkpoint in checkpoint_dir. If a path is given
        it will load the checkpoint specified by checkpoint_path.
        **Only use this method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None or checkpoint_path == "":
            ls = [str(i) for i in Path(checkpoint_dir).glob("*")]
            # example path epoch-0000_step-0017000_loss-1.972191, 找step最大的
            checkpoint_path = max(ls, key=lambda x: int(x.split("_")[-2].split("-")[-1]))

        if self.accelerator.is_main_process:
            self.logger.info("Load model from {}".format(checkpoint_path))
        print("Load model from {}".format(checkpoint_path))
        # if resume_type == "resume":
        #     self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1])
        #     self.step = int(checkpoint_path.split("_")[-2].split("-")[-1]) + 1
        #     self.accelerator.load_state(checkpoint_path)
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

    ### THIS IS MAIN ENTRY ###
    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        if self.accelerator.is_main_process:
            self.__dump_cfg(self.config_save_path)
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            # Do training & validating epoch
            train_total_loss, train_losses = self._train_epoch()
            if isinstance(train_losses, dict):
                for key, loss in train_losses.items():
                    self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            valid_total_loss, valid_losses = self._valid_epoch()
            if isinstance(valid_losses, dict):
                for key, loss in valid_losses.items():
                    self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            self.logger.info("  |- Train/Loss: {:.6f}".format(train_total_loss))
            self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_total_loss))
            self.accelerator.log(
                {
                    "Epoch/Train Loss": train_total_loss,
                    "Epoch/Valid Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()

            # Check if hit save_checkpoint_stride and run_eval
            run_eval = False
            if self.accelerator.is_main_process:
                save_checkpoint = False
                hit_dix = []
                for i, num in enumerate(self.save_checkpoint_stride):
                    if self.epoch % num == 0:
                        save_checkpoint = True
                        hit_dix.append(i)
                        run_eval |= self.run_eval[i]

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and save_checkpoint:
                path = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_total_loss
                    ),
                )
                self.accelerator.save_state(path)

                json.dump(
                    self.checkpoints_path,
                    open(os.path.join(path, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )

                # Remove old checkpoints
                to_remove = []
                for idx in hit_dix:
                    self.checkpoints_path[idx].append(path)
                    while len(self.checkpoints_path[idx]) > self.keep_last[idx]:
                        to_remove.append((idx, self.checkpoints_path[idx].pop(0)))

                # Search conflicts
                total = set()
                for i in self.checkpoints_path:
                    total |= set(i)
                do_remove = set()
                for idx, path in to_remove[::-1]:
                    if path in total:
                        self.checkpoints_path[idx].insert(0, path)
                    else:
                        do_remove.add(path)

                # Remove old checkpoints
                for path in do_remove:
                    shutil.rmtree(path, ignore_errors=True)
                    self.logger.debug(f"Remove old checkpoint: {path}")

            self.accelerator.wait_for_everyone()
            if run_eval:
                # TODO: run evaluation
                pass

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            path = os.path.join(
                self.checkpoint_dir,
                "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                    self.epoch, self.step, valid_total_loss
                ),
            )
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_total_loss
                    ),
                )
            )

            json.dump(
                self.checkpoints_path,
                open(os.path.join(path, "ckpts.json"), "w"),
                ensure_ascii=False,
                indent=4,
            )

        self.accelerator.end_training()

    ### Following are methods that can be used directly in child classes ###
    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
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
            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, _ = self._train_step(batch)
            self.batch_count += 1

            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                if isinstance(self.scheduler, dict):
                    for key in self.scheduler.keys():
                        self.scheduler[key].step()
                else:
                    if isinstance(self.scheduler, Eden):
                        self.scheduler.step_batch(self.step)
                    else:
                        self.scheduler.step()

                epoch_sum_loss += total_loss

                if isinstance(train_losses, dict):
                    for key, value in train_losses.items():
                        epoch_losses[key] += value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()

        epoch_sum_loss = (
            epoch_sum_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )

        for key in epoch_losses.keys():
            epoch_losses[key] = (
                epoch_losses[key]
                / len(self.train_dataloader)
                * self.cfg.train.gradient_accumulation_step
            )
        return epoch_sum_loss, epoch_losses

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = dict()
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
            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss += total_loss
            if isinstance(valid_losses, dict):
                for key, value in valid_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

        epoch_sum_loss = epoch_sum_loss / len(self.valid_dataloader)
        for key in epoch_losses.keys():
            epoch_losses[key] = epoch_losses[key] / len(self.valid_dataloader)

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def _train_step(self):
        pass

    def _valid_step(self, batch):
        pass

    def _inference(self):
        pass

    def _is_valid_pattern(self, directory_name):
        directory_name = str(directory_name)
        pattern = r"^epoch-\d{4}_step-\d{7}_loss-\d{1}\.\d{6}"
        return re.match(pattern, directory_name) is not None

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

    def _build_speaker_lut(self):
        # combine speakers
        if not os.path.exists(os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)):
            speakers = {}
        else:
            with open(
                os.path.join(self.exp_dir, self.cfg.preprocess.spk2id), "r"
            ) as speaker_file:
                speakers = json.load(speaker_file)
        for dataset in self.cfg.dataset:
            speaker_lut_path = os.path.join(
                self.cfg.preprocess.processed_dir, dataset, self.cfg.preprocess.spk2id
            )
            with open(speaker_lut_path, "r") as speaker_lut_path:
                singer_lut = json.load(speaker_lut_path)
            for singer in singer_lut.keys():
                if singer not in speakers:
                    speakers[singer] = len(speakers)
        with open(
            os.path.join(self.exp_dir, self.cfg.preprocess.spk2id), "w"
        ) as speaker_file:
            json.dump(speakers, speaker_file, indent=4, ensure_ascii=False)
        print(
            "speakers have been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            )
        )
        return speakers

    def _build_utt2spk_dict(self):
        # combine speakers
        utt2spk = {}
        if not os.path.exists(os.path.join(self.exp_dir, self.cfg.preprocess.utt2spk)):
            utt2spk = {}
        else:
            with open(
                os.path.join(self.exp_dir, self.cfg.preprocess.utt2spk), "r"
            ) as utt2spk_file:
                for line in utt2spk_file.readlines():
                    utt, spk = line.strip().split("\t")
                    utt2spk[utt] = spk
        for dataset in self.cfg.dataset:
            utt2spk_dict_path = os.path.join(
                self.cfg.preprocess.processed_dir, dataset, self.cfg.preprocess.utt2spk
            )
            with open(utt2spk_dict_path, "r") as utt2spk_dict:
                for line in utt2spk_dict.readlines():
                    utt, spk = line.strip().split("\t")
                    if utt not in utt2spk.keys():
                        utt2spk[utt] = spk
        with open(
            os.path.join(self.exp_dir, self.cfg.preprocess.utt2spk), "w"
        ) as utt2spk_file:
            for utt, spk in utt2spk.items():
                utt2spk_file.write(utt + "\t" + spk + "\n")
        print(
            "utterance and speaker mapper have been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.utt2spk)
            )
        )
        return utt2spk

    def _save_phone_symbols_file_to_exp_path(self):
        phone_symbols_file = os.path.join(
            self.cfg.preprocess.processed_dir,
            self.cfg.dataset[0],
            self.cfg.preprocess.symbols_dict,
        )
        phone_symbols_file_to_exp_path = os.path.join(
            self.exp_dir, self.cfg.preprocess.symbols_dict
        )
        shutil.copy(phone_symbols_file, phone_symbols_file_to_exp_path)
        print(
            "phone symbols been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.symbols_dict)
            )
        )