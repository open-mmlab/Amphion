# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from optimizer.optimizers import Eve, ScaledAdam
from schedulers.scheduler import NoamScheduler, Eden
from models.tts.valle.valle_dataset import (
    VALLEDataset,
    VALLECollator,
    batch_by_size,
)
from models.base.base_sampler import VariableSampler
from models.tts.base import TTSTrainer
from models.tts.valle.valle import VALLE
import diffusers


class VALLETrainer(TTSTrainer):
    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)

    def _build_model(self):
        model = VALLE(self.cfg.model)

        return model

    def _build_dataset(self):
        return VALLEDataset, VALLECollator

    def _build_optimizer(self):
        if self.args.train_stage:
            if isinstance(self.model, DistributedDataParallel):
                model = self.model.module
            else:
                model = self.model
            model_parameters = model.stage_parameters(self.args.train_stage)
        else:
            model_parameters = self.model.parameters()

        if self.cfg.train.optimizer == "ScaledAdam":
            parameters_names = []
            if self.args.train_stage != 0:
                parameters_names.append(
                    [
                        name_param_pair[0]
                        for name_param_pair in model.stage_named_parameters(
                            self.args.train_stage
                        )
                    ]
                )
            else:
                parameters_names.append(
                    [name_param_pair[0] for name_param_pair in model.named_parameters()]
                )

            optimizer = ScaledAdam(
                model_parameters,
                lr=self.cfg.train.base_lr,
                betas=(0.9, 0.95),
                clipping_scale=2.0,
                parameters_names=parameters_names,
                show_dominant_parameters=False,
                clipping_update_period=1000,
            )
        elif self.cfg.train.optimizer == "Eve":
            optimizer = Eve(
                model_parameters,
                lr=self.cfg.train.base_lr,
                betas=(0.9, 0.98),
                target_rms=0.1,
            )
        elif self.cfg.train.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model_parameters,
                lr=self.cfg.train.base_lr,
                betas=(0.9, 0.95),
                weight_decay=1e-2,
                eps=1e-8,
            )
        elif self.cfg.train.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model_parameters,
                lr=self.cfg.train.base_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        else:
            raise NotImplementedError()

        return optimizer

    def _build_scheduler(self):
        if self.cfg.train.scheduler.lower() == "eden":
            scheduler = Eden(
                self.optimizer, 5000, 4, warmup_batches=self.cfg.train.warmup_steps
            )
        elif self.cfg.train.scheduler.lower() == "noam":
            scheduler = NoamScheduler(
                self.cfg.train.base_lr,
                self.optimizer,
                self.cfg.model.decoder_dim,
                warmup_steps=self.cfg.train.warmup_steps,
            )
        elif self.cfg.train.scheduler.lower() == "cosine":
            from diffusers.optimization import get_cosine_schedule_with_warmup

            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.cfg.train.warmup_steps
                * self.accelerator.num_processes,
                num_training_steps=self.cfg.train.total_training_steps
                * self.accelerator.num_processes,
            )
        else:
            raise NotImplementedError(f"{self.cfg.train.scheduler}")

        return scheduler

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
                total_loss, train_losses = self._train_step(batch)
                self.accelerator.backward(total_loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.batch_count += 1

            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                if self.cfg.train.optimizer not in ["ScaledAdam", "Eve"]:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                for k in range(self.cfg.train.gradient_accumulation_step):
                    if isinstance(self.scheduler, Eden):
                        self.scheduler.step_batch(self.step)
                    else:
                        self.scheduler.step()

                epoch_sum_loss += total_loss.detach().cpu().item()

                if isinstance(train_losses, dict):
                    for key, value in train_losses.items():
                        if key not in epoch_losses.keys():
                            epoch_losses[key] = value
                        else:
                            epoch_losses[key] += value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Step/Train {}".format(key): "{:.6f}".format(loss)},
                            step=self.step,
                        )
                else:
                    self.accelerator.log(
                        {"Step/Train Loss": loss},
                        step=self.step,
                    )

                self.accelerator.log(
                    {"Step/lr": self.scheduler.get_last_lr()[0]},
                    step=self.step,
                )

                # print loss every log_epoch_step steps
                # if epoch_step % self.cfg.train.log_epoch_step == 0:
                #     for key, loss in train_losses.items():
                #         self.logger.info("Step/Train {}: {:.6f}".format(key, loss))
                #         print("Step/Train {}: {:.6f}".format(key, loss))

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

    def _train_step(self, batch, is_training=True):
        text_tokens = batch["phone_seq"].to(self.device)
        text_tokens_lens = batch["phone_len"].to(self.device)
        assert text_tokens.ndim == 2

        audio_features = batch["acoustic_token"].to(self.device)
        audio_features_lens = batch["target_len"].to(self.device)
        assert audio_features.ndim == 3

        with torch.set_grad_enabled(is_training):
            loss, losses = self.model(
                x=text_tokens,
                x_lens=text_tokens_lens,
                y=audio_features,
                y_lens=audio_features_lens,
                train_stage=self.args.train_stage,
            )

        assert loss.requires_grad == is_training

        loss_dict = {}
        frames_sum = (audio_features_lens).sum()

        avg_loss = loss / frames_sum

        loss_dict["loss"] = avg_loss.detach().cpu().item()
        for l in losses:
            loss_dict[l] = losses[l].detach().cpu().item() / frames_sum.item()

        return avg_loss, loss_dict

    def _valid_step(self, batch):
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        total_loss, valid_losses = self._train_step(
            batch=batch,
            is_training=False,
        )
        assert total_loss.requires_grad is False

        total_loss = total_loss.detach().cpu().item()

        return total_loss, valid_losses, valid_stats

    def _build_dataloader(self):
        if not self.cfg.train.use_dynamic_batchsize:
            return super()._build_dataloader()
        if len(self.cfg.dataset) > 1:
            raise Exception("use_dynamic_batchsize only supports single dataset now.")
        Dataset, Collator = self._build_dataset()
        train_dataset = Dataset(
            self.cfg, self.cfg.dataset[0], is_valid=False
        )  # TODO: support use_dynamic_batchsize for more than one datasets.
        train_collate = Collator(self.cfg)
        batch_sampler = batch_by_size(
            train_dataset.num_frame_indices,
            train_dataset.get_num_frames,
            max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
            max_sentences=self.cfg.train.max_sentences * self.accelerator.num_processes,
            required_batch_size_multiple=self.accelerator.num_processes,
        )
        np.random.seed(1234)
        np.random.shuffle(batch_sampler)
        print(batch_sampler[:1])
        batches = [
            x[self.accelerator.local_process_index :: self.accelerator.num_processes]
            for x in batch_sampler
            if len(x) % self.accelerator.num_processes == 0
        ]

        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_collate,
            num_workers=self.cfg.train.dataloader.num_worker,
            batch_sampler=VariableSampler(
                batches, drop_last=False, use_random_sampler=True
            ),
            pin_memory=False,
        )
        self.accelerator.wait_for_everyone()

        valid_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=True)
        valid_collate = Collator(self.cfg)
        batch_sampler = batch_by_size(
            valid_dataset.num_frame_indices,
            valid_dataset.get_num_frames,
            max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
            max_sentences=self.cfg.train.max_sentences * self.accelerator.num_processes,
            required_batch_size_multiple=self.accelerator.num_processes,
        )
        batches = [
            x[self.accelerator.local_process_index :: self.accelerator.num_processes]
            for x in batch_sampler
            if len(x) % self.accelerator.num_processes == 0
        ]
        valid_loader = DataLoader(
            valid_dataset,
            collate_fn=valid_collate,
            num_workers=self.cfg.train.dataloader.num_worker,
            batch_sampler=VariableSampler(batches, drop_last=False),
            pin_memory=False,
        )
        self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

    def _accelerator_prepare(self):
        if not self.cfg.train.use_dynamic_batchsize:
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

    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--train_stage",
            type=int,
            default="1",
            help="0: train all modules, 1: AR Decoder, 2: NAR Decoder",
        )
        parser.add_argument(
            "--ar_model_ckpt_dir",
            type=str,
            default=None,
            help="Checkpoint for ar model ckeckpoint in the first training stage.",
        )
