# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm

from utils.util import *
from utils.mel import mel_spectrogram_torch
from models.tts.base import TTSTrainer
from models.tts.vits.vits import SynthesizerTrn
from models.tts.vits.vits_dataset import VITSDataset, VITSCollator
from models.vocoders.gan.discriminator.mpd import (
    MultiPeriodDiscriminator_vits as MultiPeriodDiscriminator,
)


class VITSTrainer(TTSTrainer):
    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)

        if cfg.preprocess.use_spkid and cfg.train.multi_speaker_training:
            if cfg.model.n_speakers == 0:
                cfg.model.n_speaker = len(self.speakers)

    def _build_model(self):
        net_g = SynthesizerTrn(
            self.cfg.model.text_token_num,
            self.cfg.preprocess.n_fft // 2 + 1,
            self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size,
            **self.cfg.model,
        )
        net_d = MultiPeriodDiscriminator(self.cfg.model.use_spectral_norm)
        model = {"generator": net_g, "discriminator": net_d}

        return model

    def _build_dataset(self):
        return VITSDataset, VITSCollator

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

                # duration loss
                loss_dur = torch.sum(outputs_g["l_length"].float())
                loss_g["loss_dur"] = loss_dur

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
                loss_g["loss_gen_all"] = (
                    loss_dur + loss_mel + loss_kl + loss_fm + loss_gen
                )

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

        batch["linear"] = batch["linear"].transpose(2, 1)  # [b, d, t]
        batch["mel"] = batch["mel"].transpose(2, 1)  # [b, d, t]
        batch["audio"] = batch["audio"].unsqueeze(1)  # [b, d, t]

        #  Discriminator
        # Generator output
        outputs_g = self.model["generator"](batch)

        y_mel = slice_segments(
            batch["mel"],
            outputs_g["ids_slice"],
            self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size,
        )
        y_hat_mel = mel_spectrogram_torch(
            outputs_g["y_hat"].squeeze(1), self.cfg.preprocess
        )
        y = slice_segments(
            batch["audio"],
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

        batch["linear"] = batch["linear"].transpose(2, 1)  # [b, d, t]
        batch["mel"] = batch["mel"].transpose(2, 1)  # [b, d, t]
        batch["audio"] = batch["audio"].unsqueeze(1)  # [b, d, t]

        # Train Discriminator
        # Generator output
        outputs_g = self.model["generator"](batch)

        y_mel = slice_segments(
            batch["mel"],
            outputs_g["ids_slice"],
            self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size,
        )
        y_hat_mel = mel_spectrogram_torch(
            outputs_g["y_hat"].squeeze(1), self.cfg.preprocess
        )
        y = slice_segments(
            batch["audio"],
            outputs_g["ids_slice"] * self.cfg.preprocess.hop_size,
            self.cfg.preprocess.segment_size,
        )

        # Discriminator output
        outputs_d = self.model["discriminator"](y, outputs_g["y_hat"].detach())
        ##  Discriminator loss
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
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1

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
