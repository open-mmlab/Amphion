from models.base.base_trainer import BaseTrainer
from models.sgmse.dereverberation.dereverberation_dataset import Specs
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from utils.sgmse_util.inference import evaluate_model
from torch_ema import ExponentialMovingAverage
from models.sgmse.dereverberation.dereverberation import ScoreModel
from modules.sgmse import sampling
from torch.utils.data import DataLoader
import os


class DereverberationTrainer(BaseTrainer):
    def __init__(self, args, cfg):
        BaseTrainer.__init__(self, args, cfg)
        self.cfg = cfg
        self.save_config_file()
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=self.cfg.train.ema_decay
        )
        self._error_loading_ema = False
        self.t_eps = self.cfg.train.t_eps
        self.num_eval_files = self.cfg.train.num_eval_files
        self.data_loader = self.build_data_loader()
        self.save_config_file()

        checkpoint = self.load_checkpoint()
        if checkpoint:
            self.load_model(checkpoint)

    def build_dataset(self):
        return Specs

    def load_checkpoint(self):
        model_path = self.cfg.train.checkpoint
        if not model_path or not os.path.exists(model_path):
            self.logger.info("No checkpoint to load or checkpoint path does not exist.")
            return None
        if not self.cfg.train.ddp or self.args.local_rank == 0:
            self.logger.info(f"Re(store) from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        if "ema" in checkpoint:
            try:
                self.ema.load_state_dict(checkpoint["ema"])
            except:
                self._error_loading_ema = True
                warnings.warn("EMA state_dict not found in checkpoint!")
        return checkpoint

    def build_data_loader(self):
        Dataset = self.build_dataset()
        train_set = Dataset(self.cfg, subset="train", shuffle_spec=True)
        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True,
        )
        self.valid_set = Dataset(self.cfg, subset="valid", shuffle_spec=False)
        valid_loader = DataLoader(
            self.valid_set,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=False,
        )
        data_loader = {"train": train_loader, "valid": valid_loader}
        return data_loader

    def build_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfg.train.adam)
        return optimizer

    def build_scheduler(self):
        return None
        # return ReduceLROnPlateau(self.optimizer["opt_ae"], **self.cfg.train.lronPlateau)

    def build_singers_lut(self):
        return None

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def _loss(self, err):
        losses = torch.square(err.abs())
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def build_criterion(self):
        return self._loss

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
            "ema": self.ema.state_dict(),
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])

    def build_model(self):
        self.model = ScoreModel(self.cfg.model.sgmse)
        return self.model

    def get_pc_sampler(
        self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs
    ):
        N = self.model.sde.N if N is None else N
        sde = self.model.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name,
                corrector_name,
                sde=sde,
                score_fn=self.model,
                y=y,
                **kwargs,
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name,
                        corrector_name,
                        sde=sde,
                        score_fn=self.model,
                        y=y_mini,
                        **kwargs,
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def _step(self, batch):
        x = batch["X"]
        y = batch["Y"]

        t = (
            torch.rand(x.shape[0], device=x.device) * (self.model.sde.T - self.t_eps)
            + self.t_eps
        )
        mean, std = self.model.sde.marginal_prob(x, t, y)

        z = torch.randn_like(x)
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        score = self.model(perturbed_data, t, y)

        err = score * sigmas + z
        loss = self.criterion(err)
        return loss

    def train_step(self, batch):
        loss = self._step(batch)

        # Backward pass and optimization
        self.optimizer.zero_grad()  # reset gradient
        loss.backward()
        self.optimizer.step()

        # Update the EMA of the model parameters
        self.ema.update(self.model.parameters())

        self.write_summary({"train_loss": loss.item()}, {})
        return {"train_loss": loss.item()}, {}, loss.item()

    def eval_step(self, batch, batch_idx):
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        loss = self._step(batch)
        self.write_valid_summary({"valid_loss": loss.item()}, {})
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.write_valid_summary(
                {"pesq": pesq, "si_sdr": si_sdr, "estoi": estoi}, {}
            )
            print(f" pesq={pesq}, si_sdr={si_sdr}, estoi={estoi}")
        if self.ema.collected_params is not None:
            self.ema.restore(self.model.parameters())
        return {"valid_loss": loss.item()}, {}, loss.item()
