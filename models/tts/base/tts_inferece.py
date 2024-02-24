# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import time
import accelerate
import random
import numpy as np
from tqdm import tqdm
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from safetensors.torch import load_file


from abc import abstractmethod
from pathlib import Path
from utils.io import save_audio
from utils.util import load_config
from models.vocoders.vocoder_inference import synthesis


class TTSInference(object):
    def __init__(self, args=None, cfg=None):
        super().__init__()

        start = time.monotonic_ns()
        self.args = args
        self.cfg = cfg
        self.infer_type = args.mode

        # get exp_dir
        if self.args.acoustics_dir is not None:
            self.exp_dir = self.args.acoustics_dir
        elif self.args.checkpoint_path is not None:
            self.exp_dir = os.path.dirname(os.path.dirname(self.args.checkpoint_path))

        # Init accelerator
        self.accelerator = accelerate.Accelerator()
        self.accelerator.wait_for_everyone()
        self.device = self.accelerator.device

        # Get logger
        with self.accelerator.main_process_first():
            self.logger = get_logger("inference", log_level=args.log_level)

        # Log some info
        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New inference process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")

        self.acoustic_model_dir = args.acoustics_dir
        self.logger.debug(f"Acoustic model dir: {args.acoustics_dir}")

        if args.vocoder_dir is not None:
            self.vocoder_dir = args.vocoder_dir
            self.logger.debug(f"Vocoder dir: {args.vocoder_dir}")

        os.makedirs(args.output_dir, exist_ok=True)

        # Set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # Setup data loader
        if self.infer_type == "batch":
            with self.accelerator.main_process_first():
                self.logger.info("Building dataset...")
                start = time.monotonic_ns()
                self.test_dataloader = self._build_test_dataloader()
                end = time.monotonic_ns()
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )

        # Build model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            self.logger.info(f"Building model done in {(end - start) / 1e6:.3f}ms")

        # Init with accelerate
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        self.accelerator = accelerate.Accelerator()
        self.model = self.accelerator.prepare(self.model)
        if self.infer_type == "batch":
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
        end = time.monotonic_ns()
        self.accelerator.wait_for_everyone()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.3f}ms")

        with self.accelerator.main_process_first():
            self.logger.info("Loading checkpoint...")
            start = time.monotonic_ns()
            if args.acoustics_dir is not None:
                self._load_model(
                    checkpoint_dir=os.path.join(args.acoustics_dir, "checkpoint")
                )
            elif args.checkpoint_path is not None:
                self._load_model(checkpoint_path=args.checkpoint_path)
            else:
                print("Either checkpoint dir or checkpoint path should be provided.")

            end = time.monotonic_ns()
            self.logger.info(f"Loading checkpoint done in {(end - start) / 1e6:.3f}ms")

        self.model.eval()
        self.accelerator.wait_for_everyone()

    def _build_test_dataset(self):
        pass

    def _build_model(self):
        pass

    # TODO: LEGACY CODE
    def _build_test_dataloader(self):
        datasets, collate = self._build_test_dataset()
        self.test_dataset = datasets(self.args, self.cfg)
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

    def _load_model(
        self,
        checkpoint_dir: str = None,
        checkpoint_path: str = None,
        old_mode: bool = False,
    ):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """

        if checkpoint_path is None:
            assert checkpoint_dir is not None
            # Load the latest accelerator state dicts
            ls = [
                str(i) for i in Path(checkpoint_dir).glob("*") if not "audio" in str(i)
            ]
            ls.sort(key=lambda x: int(x.split("_")[-3].split("-")[-1]), reverse=True)
            checkpoint_path = ls[0]

        if (
            Path(os.path.join(checkpoint_path, "model.safetensors")).exists()
            and accelerate.__version__ < "0.25"
        ):
            self.model.load_state_dict(
                load_file(os.path.join(checkpoint_path, "model.safetensors")),
                strict=False,
            )
        else:
            self.accelerator.load_state(str(checkpoint_path))
        return str(checkpoint_path)

    def inference(self):
        if self.infer_type == "single":
            out_dir = os.path.join(self.args.output_dir, "single")
            os.makedirs(out_dir, exist_ok=True)

            pred_audio = self.inference_for_single_utterance()
            save_path = os.path.join(out_dir, "test_pred.wav")
            save_audio(save_path, pred_audio, self.cfg.preprocess.sample_rate)

        elif self.infer_type == "batch":
            out_dir = os.path.join(self.args.output_dir, "batch")
            os.makedirs(out_dir, exist_ok=True)

            pred_audio_list = self.inference_for_batches()
            for it, wav in zip(self.test_dataset.metadata, pred_audio_list):
                uid = it["Uid"]
                save_audio(
                    os.path.join(out_dir, f"{uid}.wav"),
                    wav.numpy(),
                    self.cfg.preprocess.sample_rate,
                    add_silence=True,
                    turn_up=True,
                )
                tmp_file = os.path.join(out_dir, f"{uid}.pt")
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
        print("Saved to: ", out_dir)

    @torch.inference_mode()
    def inference_for_batches(self):
        y_pred = []
        for i, batch in tqdm(enumerate(self.test_dataloader)):
            y_pred, mel_lens, _ = self._inference_each_batch(batch)
            y_ls = y_pred.chunk(self.test_batch_size)
            tgt_ls = mel_lens.chunk(self.test_batch_size)
            j = 0
            for it, l in zip(y_ls, tgt_ls):
                l = l.item()
                it = it.squeeze(0)[:l].detach().cpu()

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
                    os.path.join(self.args.output_dir, "{}.pt".format(item["Uid"]))
                ).numpy()
                for item in self.test_dataset.metadata
            ],
        )
        for it, wav in zip(self.test_dataset.metadata, res):
            uid = it["Uid"]
            save_audio(
                os.path.join(self.args.output_dir, f"{uid}.wav"),
                wav.numpy(),
                22050,
                add_silence=True,
                turn_up=True,
            )

    @abstractmethod
    @torch.inference_mode()
    def _inference_each_batch(self, batch_data):
        pass

    def inference_for_single_utterance(self, text):
        pass

    def synthesis_by_vocoder(self, pred):
        audios_pred = synthesis(
            self.vocoder_cfg,
            self.checkpoint_dir_vocoder,
            len(pred),
            pred,
        )

        return audios_pred

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

    def _set_random_seed(self, seed):
        """Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
