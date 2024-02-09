# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import json5
from tqdm import tqdm
import json
import shutil

from models.svc.base import SVCTrainer
from modules.encoder.condition_encoder import ConditionEncoder
from models.svc.comosvc.comosvc import ComoSVC


class ComoSVCTrainer(SVCTrainer):
    r"""The base trainer for all diffusion models. It inherits from SVCTrainer and
    implements ``_build_model`` and ``_forward_step`` methods.
    """

    def __init__(self, args=None, cfg=None):
        SVCTrainer.__init__(self, args, cfg)
        self.distill = cfg.model.comosvc.distill
        self.skip_diff = True

    ### Following are methods only for comoSVC models ###

    def _load_teacher_model(self, model):
        r"""Load teacher model from checkpoint file."""
        self.checkpoint_file = self.teacher_model_path
        self.logger.info(
            "Load teacher acoustic model from {}".format(self.checkpoint_file)
        )
        raw_dict = torch.load(self.checkpoint_file)
        model.load_state_dict(raw_dict)

    def _build_model(self):
        r"""Build the model for training. This function is called in ``__init__`` function."""

        # TODO: sort out the config
        self.cfg.model.condition_encoder.f0_min = self.cfg.preprocess.f0_min
        self.cfg.model.condition_encoder.f0_max = self.cfg.preprocess.f0_max
        self.condition_encoder = ConditionEncoder(self.cfg.model.condition_encoder)
        self.acoustic_mapper = ComoSVC(self.cfg)
        model = torch.nn.ModuleList([self.condition_encoder, self.acoustic_mapper])
        if self.cfg.model.comosvc.distill:
            if not self.args.resume:
                # do not load teacher model when resume
                self.teacher_model_path = self.cfg.model.teacher_model_path
                self._load_teacher_model(model)
            # build teacher & target decoder and freeze teacher
            self.acoustic_mapper.decoder.init_consistency_training()
            self.freeze_net(self.condition_encoder)
            self.freeze_net(self.acoustic_mapper.encoder)
            self.freeze_net(self.acoustic_mapper.decoder.denoise_fn_pretrained)
            self.freeze_net(self.acoustic_mapper.decoder.denoise_fn_ema)
        return model

    def freeze_net(self, model):
        r"""Freeze the model for training."""
        for name, param in model.named_parameters():
            param.requires_grad = False

    def __build_optimizer(self):
        r"""Build optimizer for training. This function is called in ``__init__`` function."""

        if self.cfg.train.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params=filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.cfg.train.adamw,
            )

        else:
            raise NotImplementedError(
                "Not support optimizer: {}".format(self.cfg.train.optimizer)
            )

        return optimizer

    def _forward_step(self, batch):
        r"""Forward step for training and inference. This function is called
        in ``_train_step`` & ``_test_step`` function.
        """
        loss = {}
        mask = batch["mask"]
        mel_input = batch["mel"]
        cond = self.condition_encoder(batch)
        if self.distill:
            cond = cond.detach()
        self.skip_diff = True if self.step < self.cfg.train.fast_steps else False
        ssim_loss, prior_loss, diff_loss = self.acoustic_mapper.compute_loss(
            mask, cond, mel_input, skip_diff=self.skip_diff
        )
        if self.distill:
            loss["distil_loss"] = diff_loss
        else:
            loss["ssim_loss_encoder"] = ssim_loss
            loss["prior_loss_encoder"] = prior_loss
            loss["diffusion_loss_decoder"] = diff_loss

        return loss

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.train()
        epoch_sum_loss: float = 0.0
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
                loss = self._train_step(batch)
                total_loss = 0
                for k, v in loss.items():
                    total_loss += v
                self.accelerator.backward(total_loss)
                enc_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.acoustic_mapper.encoder.parameters(), max_norm=1
                )
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.acoustic_mapper.decoder.parameters(), max_norm=1
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.batch_count += 1

            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += total_loss
                log_info = {}
                for k, v in loss.items():
                    key = "Step/Train Loss/{}".format(k)
                    log_info[key] = v
                log_info["Step/Learning Rate"] = self.optimizer.param_groups[0]["lr"]
                self.accelerator.log(
                    log_info,
                    step=self.step,
                )
                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()
        return (
            epoch_sum_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step,
            loss,
        )

    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        if self.accelerator.is_main_process:
            self.__dump_cfg(self.config_save_path)
        self.model.train()
        self.optimizer.zero_grad()
        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            ### TODO: change the return values of _train_epoch() to a loss dict, or (total_loss, loss_dict)
            ### It's inconvenient for the model with multiple losses
            # Do training & validating epoch
            train_loss, loss = self._train_epoch()
            self.logger.info("  |- Train/Loss: {:.6f}".format(train_loss))
            for k, v in loss.items():
                self.logger.info("  |- Train/Loss/{}: {:.6f}".format(k, v))
            valid_loss = self._valid_epoch()
            self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_loss))
            self.accelerator.log(
                {"Epoch/Train Loss": train_loss, "Epoch/Valid Loss": valid_loss},
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()
            # TODO: what is scheduler?
            self.scheduler.step(valid_loss)  # FIXME: use epoch track correct?

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
            if (
                self.accelerator.is_main_process
                and save_checkpoint
                and (self.distill or not self.skip_diff)
            ):
                path = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_loss
                    ),
                )
                self.tmp_checkpoint_save_path = path
                self.accelerator.save_state(path)
                print(f"save checkpoint in {path}")
                json.dump(
                    self.checkpoints_path,
                    open(os.path.join(path, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
                self._save_auxiliary_states()

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
            for k, v in batch_loss.items():
                epoch_sum_loss += v

        self.accelerator.wait_for_everyone()
        return epoch_sum_loss / len(self.valid_dataloader)

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
