# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import re
import time
from abc import abstractmethod
from pathlib import Path

import accelerate
import json5
import numpy as np
import torch
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from models.vocoders.vocoder_inference import synthesis
from utils.io import save_audio
from utils.util import load_config
from utils.audio_slicer import is_silence

EPS = 1.0e-12


class BaseInference(object):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        super().__init__()

        start = time.monotonic_ns()
        self.args = args
        self.cfg = cfg

        assert infer_type in ["from_dataset", "from_file"]
        self.infer_type = infer_type

        # init with accelerate
        self.accelerator = accelerate.Accelerator()
        self.accelerator.wait_for_everyone()

        # Use accelerate logger for distributed inference
        with self.accelerator.main_process_first():
            self.logger = get_logger("inference", log_level=args.log_level)

        # Log some info
        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New inference process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")
        self.logger.debug(f"Using {args.log_level.upper()} logging level.")

        self.acoustics_dir = args.acoustics_dir
        self.logger.debug(f"Acoustic dir: {args.acoustics_dir}")
        self.vocoder_dir = args.vocoder_dir
        self.logger.debug(f"Vocoder dir: {args.vocoder_dir}")
        # should be in svc inferencer
        # self.target_singer = args.target_singer
        # self.logger.info(f"Target singers: {args.target_singer}")
        # self.trans_key = args.trans_key
        # self.logger.info(f"Trans key: {args.trans_key}")

        os.makedirs(args.output_dir, exist_ok=True)

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.test_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # setup model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            # self.logger.debug(self.model)
            self.logger.info(f"Building model done in {(end - start) / 1e6:.3f}ms")

        # init with accelerate
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        self.accelerator = accelerate.Accelerator()
        self.model = self.accelerator.prepare(self.model)
        end = time.monotonic_ns()
        self.accelerator.wait_for_everyone()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.3f}ms")

        with self.accelerator.main_process_first():
            self.logger.info("Loading checkpoint...")
            start = time.monotonic_ns()
            # TODO: Also, suppose only use latest one yet
            self.__load_model(os.path.join(args.acoustics_dir, "checkpoint"))
            end = time.monotonic_ns()
            self.logger.info(f"Loading checkpoint done in {(end - start) / 1e6:.3f}ms")

        self.model.eval()
        self.accelerator.wait_for_everyone()

    ### Abstract methods ###
    @abstractmethod
    def _build_test_dataset(self):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    @torch.inference_mode()
    def _inference_each_batch(self, batch_data):
        pass

    ### Abstract methods end ###

    @torch.inference_mode()
    def inference(self):
        for i, batch in enumerate(self.test_dataloader):
            y_pred = self._inference_each_batch(batch).cpu()

            # Judge whether the min-max normliazation is used
            if self.cfg.preprocess.use_min_max_norm_mel:
                mel_min, mel_max = self.test_dataset.target_mel_extrema
                y_pred = (y_pred + 1.0) / 2.0 * (mel_max - mel_min + EPS) + mel_min

            y_ls = y_pred.chunk(self.test_batch_size)
            tgt_ls = batch["target_len"].cpu().chunk(self.test_batch_size)
            j = 0
            for it, l in zip(y_ls, tgt_ls):
                l = l.item()
                it = it.squeeze(0)[:l]
                uid = self.test_dataset.metadata[i * self.test_batch_size + j]["Uid"]
                torch.save(it, os.path.join(self.args.output_dir, f"{uid}.pt"))
                j += 1

        vocoder_cfg, vocoder_ckpt = self._parse_vocoder(self.args.vocoder_dir)

        res = synthesis(
            cfg=vocoder_cfg,
            vocoder_weight_file=vocoder_ckpt,
            n_samples=None,
            pred=[
                torch.load(
                    os.path.join(self.args.output_dir, "{}.pt".format(i["Uid"]))
                ).numpy(force=True)
                for i in self.test_dataset.metadata
            ],
        )

        output_audio_files = []
        for it, wav in zip(self.test_dataset.metadata, res):
            uid = it["Uid"]
            file = os.path.join(self.args.output_dir, f"{uid}.wav")
            output_audio_files.append(file)

            wav = wav.numpy(force=True)
            save_audio(
                file,
                wav,
                self.cfg.preprocess.sample_rate,
                add_silence=False,
                turn_up=not is_silence(wav, self.cfg.preprocess.sample_rate),
            )
            os.remove(os.path.join(self.args.output_dir, f"{uid}.pt"))

        return sorted(output_audio_files)

    # TODO: LEGACY CODE
    def _build_dataloader(self):
        datasets, collate = self._build_test_dataset()
        self.test_dataset = datasets(self.args, self.cfg, self.infer_type)
        self.test_collate = collate(self.cfg)
        self.test_batch_size = min(
            self.cfg.train.batch_size, len(self.test_dataset.metadata)
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            collate_fn=self.test_collate,
            num_workers=1,
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        return test_dataloader

    def __load_model(self, checkpoint_dir: str = None, checkpoint_path: str = None):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            ls = []
            for i in Path(checkpoint_dir).iterdir():
                if re.match(r"epoch-\d+_step-\d+_loss-[\d.]+", str(i.stem)):
                    ls.append(i)
            ls.sort(
                key=lambda x: int(x.stem.split("_")[-3].split("-")[-1]), reverse=True
            )
            checkpoint_path = ls[0]
        else:
            checkpoint_path = Path(checkpoint_path)
        self.accelerator.load_state(str(checkpoint_path))
        # set epoch and step
        self.epoch = int(checkpoint_path.stem.split("_")[-3].split("-")[-1])
        self.step = int(checkpoint_path.stem.split("_")[-2].split("-")[-1])
        return str(checkpoint_path)

    @staticmethod
    def _set_random_seed(seed):
        r"""Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    @staticmethod
    def _parse_vocoder(vocoder_dir):
        r"""Parse vocoder config"""
        vocoder_dir = os.path.abspath(vocoder_dir)
        ckpt_list = [ckpt for ckpt in Path(vocoder_dir).glob("*.pt")]
        ckpt_list.sort(key=lambda x: int(x.stem), reverse=True)
        ckpt_path = str(ckpt_list[0])
        vocoder_cfg = load_config(
            os.path.join(vocoder_dir, "args.json"), lowercase=True
        )
        return vocoder_cfg, ckpt_path

    @staticmethod
    def __count_parameters(model):
        return sum(p.numel() for p in model.parameters())

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
