# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from pathlib import Path

import accelerate

# from models.svc.base import SVCTrainer
from models.vc.base.vc_dataset import VCCollator, VCDataset
from models.vc.vits.vits import *
from models.tts.base import TTSTrainer

from utils.mel import mel_spectrogram_torch
import json

from models.vocoders.gan.discriminator.mpd import (
    MultiPeriodDiscriminator_vits as MultiPeriodDiscriminator,
)


class VitsVCTrainer(TTSTrainer):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self._init_accelerator()

        TTSTrainer.__init__(self, args, cfg)

    def _build_model(self):
        net_g = SynthesizerTrn(
            self.cfg.preprocess.n_fft // 2 + 1,
            self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size,
            # directly use cfg
            self.cfg,
        )
        net_d = MultiPeriodDiscriminator(self.cfg.model.vits.use_spectral_norm)
        model = {"generator": net_g, "discriminator": net_d}

        return model

    def _build_dataset(self):
        return VCDataset, VCCollator

    def _build_optimizer(self):
        optimizer_g = torch.optim.AdamW(
            self.model["generator"].parameters(),
            self.cfg.train.learning_rate,
            betas=self.cfg.train.AdamW.betas,
            eps=self.cfg.train.AdamW.eps,
        )
        optimizer_d = torch.optim.AdamW(
            self.model["discriminator"].parameters(),
            self.cfg.train.learning_rate,
            betas=self.cfg.train.AdamW.betas,
            eps=self.cfg.train.AdamW.eps,
        )
        optimizer = {"optimizer_g": optimizer_g, "optimizer_d": optimizer_d}

        return optimizer

    def _build_scheduler(self):
        scheduler_g = ExponentialLR(
            self.optimizer["optimizer_g"],
            gamma=self.cfg.train.lr_decay,
            last_epoch=self.epoch - 1,
        )
        scheduler_d = ExponentialLR(
            self.optimizer["optimizer_d"],
            gamma=self.cfg.train.lr_decay,
            last_epoch=self.epoch - 1,
        )

        scheduler = {"scheduler_g": scheduler_g, "scheduler_d": scheduler_d}
        return scheduler

    def _build_criterion(self):
        class GeneratorLoss(nn.Module):
            def __init__(self, cfg):
                super(GeneratorLoss, self).__init__()
                self.cfg = cfg
                self.l1_loss = nn.L1Loss()

            def generator_loss(self, disc_outputs):
                loss = 0
                gen_losses = []
                for dg in disc_outputs:
                    dg = dg.float()
                    l = torch.mean((1 - dg) ** 2)
                    gen_losses.append(l)
                    loss += l

                return loss, gen_losses

            def feature_loss(self, fmap_r, fmap_g):
                loss = 0
                for dr, dg in zip(fmap_r, fmap_g):
                    for rl, gl in zip(dr, dg):
                        rl = rl.float().detach()
                        gl = gl.float()
                        loss += torch.mean(torch.abs(rl - gl))

                return loss * 2

            def kl_loss(self, z_p, logs_q, m_p, logs_p, z_mask):
                """
                z_p, logs_q: [b, h, t_t]
                m_p, logs_p: [b, h, t_t]
                """
                z_p = z_p.float()
                logs_q = logs_q.float()
                m_p = m_p.float()
                logs_p = logs_p.float()
                z_mask = z_mask.float()

                kl = logs_p - logs_q - 0.5
                kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
                kl = torch.sum(kl * z_mask)
                l = kl / torch.sum(z_mask)
                return l

            def forward(
                self,
                outputs_g,
                outputs_d,
                y_mel,
                y_hat_mel,
            ):
                loss_g = {}

                # mel loss
                loss_mel = self.l1_loss(y_mel, y_hat_mel) * self.cfg.train.c_mel
                loss_g["loss_mel"] = loss_mel

                # kl loss
                loss_kl = (
                    self.kl_loss(
                        outputs_g["z_p"],
                        outputs_g["logs_q"],
                        outputs_g["m_p"],
                        outputs_g["logs_p"],
                        outputs_g["z_mask"],
                    )
                    * self.cfg.train.c_kl
                )
                loss_g["loss_kl"] = loss_kl

                # feature loss
                loss_fm = self.feature_loss(outputs_d["fmap_rs"], outputs_d["fmap_gs"])
                loss_g["loss_fm"] = loss_fm

                # gan loss
                loss_gen, losses_gen = self.generator_loss(outputs_d["y_d_hat_g"])
                loss_g["loss_gen"] = loss_gen
                loss_g["loss_gen_all"] = loss_mel + loss_kl + loss_fm + loss_gen

                return loss_g

        class DiscriminatorLoss(nn.Module):
            def __init__(self, cfg):
                super(DiscriminatorLoss, self).__init__()
                self.cfg = cfg
                self.l1Loss = torch.nn.L1Loss(reduction="mean")

            def __call__(self, disc_real_outputs, disc_generated_outputs):
                loss_d = {}

                loss = 0
                r_losses = []
                g_losses = []
                for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
                    dr = dr.float()
                    dg = dg.float()
                    r_loss = torch.mean((1 - dr) ** 2)
                    g_loss = torch.mean(dg**2)
                    loss += r_loss + g_loss
                    r_losses.append(r_loss.item())
                    g_losses.append(g_loss.item())

                loss_d["loss_disc_all"] = loss

                return loss_d

        criterion = {
            "generator": GeneratorLoss(self.cfg),
            "discriminator": DiscriminatorLoss(self.cfg),
        }
        return criterion

    def _check_resume(self):
        if self.args.resume:
            if self.args.resume_from_ckpt_path == "":
                ## Automatically resume according to the current exprimental name
                self.logger.info(
                    "Automatically resuming from latest checkpoint in {}...".format(
                        self.checkpoint_dir
                    )
                )
                start = time.monotonic_ns()
                ckpt_path = self.__load_model(
                    checkpoint_dir=self.checkpoint_dir,
                    resume_type=self.args.resume_type,
                )
                end = time.monotonic_ns()
                self.logger.info(
                    f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                )
                self.checkpoints_path = json.load(
                    open(os.path.join(ckpt_path, "ckpts.json"), "r")
                )
            else:
                ## Resume from the given checkpoint path
                if not os.path.exists(self.args.resume_from_ckpt_path):
                    raise ValueError(
                        "[Error] The resumed checkpoint path {} don't exist.".format(
                            self.args.resume_from_ckpt_path
                        )
                    )
                self.logger.info(
                    "Resuming from {}...".format(self.args.resume_from_ckpt_path)
                )
                start = time.monotonic_ns()
                ckpt_path = self.__load_model(
                    checkpoint_path=self.args.resume_from_ckpt_path,
                    resume_type=self.args.resume_type,
                )
                end = time.monotonic_ns()
                self.logger.info(
                    f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                )

    def __load_model(
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
            ls = [str(i) for i in Path(checkpoint_dir).glob("*")]
            ls.sort(key=lambda x: int(x.split("_")[-3].split("-")[-1]), reverse=True)
            checkpoint_path = ls[0]
            self.logger.info("Resume from {}...".format(checkpoint_path))

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

    # Keep legacy unchanged
    def write_summary(
        self,
        losses,
        stats,
        images={},
        audios={},
        audio_sampling_rate=24000,
        tag="train",
    ):
        for key, value in losses.items():
            self.sw.add_scalar(tag + "/" + key, value, self.step)
        self.sw.add_scalar(
            "learning_rate",
            self.optimizer["optimizer_g"].param_groups[0]["lr"],
            self.step,
        )

        if len(images) != 0:
            for key, value in images.items():
                self.sw.add_image(key, value, self.global_step, batchformats="HWC")
        if len(audios) != 0:
            for key, value in audios.items():
                self.sw.add_audio(key, value, self.global_step, audio_sampling_rate)

    def write_valid_summary(
        self, losses, stats, images={}, audios={}, audio_sampling_rate=24000, tag="val"
    ):
        for key, value in losses.items():
            self.sw.add_scalar(tag + "/" + key, value, self.step)

        if len(images) != 0:
            for key, value in images.items():
                self.sw.add_image(key, value, self.global_step, batchformats="HWC")
        if len(audios) != 0:
            for key, value in audios.items():
                self.sw.add_audio(key, value, self.global_step, audio_sampling_rate)

    def _get_state_dict(self):
        state_dict = {
            "generator": self.model["generator"].state_dict(),
            "discriminator": self.model["discriminator"].state_dict(),
            "optimizer_g": self.optimizer["optimizer_g"].state_dict(),
            "optimizer_d": self.optimizer["optimizer_d"].state_dict(),
            "scheduler_g": self.scheduler["scheduler_g"].state_dict(),
            "scheduler_d": self.scheduler["scheduler_d"].state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def get_state_dict(self):
        state_dict = {
            "generator": self.model["generator"].state_dict(),
            "discriminator": self.model["discriminator"].state_dict(),
            "optimizer_g": self.optimizer["optimizer_g"].state_dict(),
            "optimizer_d": self.optimizer["optimizer_d"].state_dict(),
            "scheduler_g": self.scheduler["scheduler_g"].state_dict(),
            "scheduler_d": self.scheduler["scheduler_d"].state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.model["generator"].load_state_dict(checkpoint["generator"])
        self.model["discriminator"].load_state_dict(checkpoint["discriminator"])
        self.optimizer["optimizer_g"].load_state_dict(checkpoint["optimizer_g"])
        self.optimizer["optimizer_d"].load_state_dict(checkpoint["optimizer_d"])
        self.scheduler["scheduler_g"].load_state_dict(checkpoint["scheduler_g"])
        self.scheduler["scheduler_d"].load_state_dict(checkpoint["scheduler_d"])

    @torch.inference_mode()
    def _valid_step(self, batch):
        r"""Testing forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_test_epoch`` for usage.
        """

        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        #  Discriminator
        # Generator output
        outputs_g = self.model["generator"](batch)

        y_mel = slice_segments(
            batch["mel"].transpose(1, 2),
            outputs_g["ids_slice"],
            self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size,
        )
        y_hat_mel = mel_spectrogram_torch(
            outputs_g["y_hat"].squeeze(1), self.cfg.preprocess
        )
        y = slice_segments(
            batch["audio"].unsqueeze(1),
            outputs_g["ids_slice"] * self.cfg.preprocess.hop_size,
            self.cfg.preprocess.segment_size,
        )

        # Discriminator output
        outputs_d = self.model["discriminator"](y, outputs_g["y_hat"].detach())
        ##  Discriminator loss
        loss_d = self.criterion["discriminator"](
            outputs_d["y_d_hat_r"], outputs_d["y_d_hat_g"]
        )
        valid_losses.update(loss_d)

        ##  Generator
        outputs_d = self.model["discriminator"](y, outputs_g["y_hat"])
        loss_g = self.criterion["generator"](outputs_g, outputs_d, y_mel, y_hat_mel)
        valid_losses.update(loss_g)

        for item in valid_losses:
            valid_losses[item] = valid_losses[item].item()

        total_loss = loss_g["loss_gen_all"] + loss_d["loss_disc_all"]

        return (
            total_loss.item(),
            valid_losses,
            valid_stats,
        )

    def _train_step(self, batch):
        r"""Forward step for training and inference. This function is called
        in ``_train_step`` & ``_test_step`` function.
        """

        train_losses = {}
        total_loss = 0
        training_stats = {}

        ## Train Discriminator
        # Generator output
        outputs_g = self.model["generator"](batch)

        y_mel = slice_segments(
            batch["mel"].transpose(1, 2),
            outputs_g["ids_slice"],
            self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size,
        )
        y_hat_mel = mel_spectrogram_torch(
            outputs_g["y_hat"].squeeze(1), self.cfg.preprocess
        )

        y = slice_segments(
            # [1, 168418] -> [1, 1, 168418]
            batch["audio"].unsqueeze(1),
            outputs_g["ids_slice"] * self.cfg.preprocess.hop_size,
            self.cfg.preprocess.segment_size,
        )

        # Discriminator output
        outputs_d = self.model["discriminator"](y, outputs_g["y_hat"].detach())
        #  Discriminator loss
        loss_d = self.criterion["discriminator"](
            outputs_d["y_d_hat_r"], outputs_d["y_d_hat_g"]
        )
        train_losses.update(loss_d)

        # BP and Grad Updated
        self.optimizer["optimizer_d"].zero_grad()
        self.accelerator.backward(loss_d["loss_disc_all"])
        self.optimizer["optimizer_d"].step()

        ## Train Generator
        outputs_d = self.model["discriminator"](y, outputs_g["y_hat"])
        loss_g = self.criterion["generator"](outputs_g, outputs_d, y_mel, y_hat_mel)
        train_losses.update(loss_g)

        # BP and Grad Updated
        self.optimizer["optimizer_g"].zero_grad()
        self.accelerator.backward(loss_g["loss_gen_all"])
        self.optimizer["optimizer_g"].step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        total_loss = loss_g["loss_gen_all"] + loss_d["loss_disc_all"]

        return (
            total_loss.item(),
            train_losses,
            training_stats,
        )

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
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
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1

            # Update info for each step
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += total_loss
                for key, value in train_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

                self.accelerator.log(
                    {
                        "Step/Generator Loss": train_losses["loss_gen_all"],
                        "Step/Discriminator Loss": train_losses["loss_disc_all"],
                        "Step/Generator Learning Rate": self.optimizer[
                            "optimizer_d"
                        ].param_groups[0]["lr"],
                        "Step/Discriminator Learning Rate": self.optimizer[
                            "optimizer_g"
                        ].param_groups[0]["lr"],
                    },
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

    def _build_singer_lut(self):
        resumed_singer_path = None
        if self.args.resume_from_ckpt_path and self.args.resume_from_ckpt_path != "":
            resumed_singer_path = os.path.join(
                self.args.resume_from_ckpt_path, self.cfg.preprocess.spk2id
            )
        if os.path.exists(os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)):
            resumed_singer_path = os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)

        if resumed_singer_path:
            with open(resumed_singer_path, "r") as f:
                singers = json.load(f)
        else:
            singers = dict()

        for dataset in self.cfg.dataset:
            singer_lut_path = os.path.join(
                self.cfg.preprocess.processed_dir, dataset, self.cfg.preprocess.spk2id
            )
            with open(singer_lut_path, "r") as singer_lut_path:
                singer_lut = json.load(singer_lut_path)
            for singer in singer_lut.keys():
                if singer not in singers:
                    singers[singer] = len(singers)

        with open(
            os.path.join(self.exp_dir, self.cfg.preprocess.spk2id), "w"
        ) as singer_file:
            json.dump(singers, singer_file, indent=4, ensure_ascii=False)
        print(
            "singers have been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            )
        )
        return singers
