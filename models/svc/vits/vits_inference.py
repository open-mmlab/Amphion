# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models.svc.base import SVCInference
from models.svc.vits.vits import SynthesizerTrn

from models.svc.base.svc_dataset import SVCTestDataset, SVCTestCollator
from utils.io import save_audio
from utils.audio_slicer import is_silence


class VitsInference(SVCInference):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        SVCInference.__init__(self, args, cfg)

    def _build_model(self):
        net_g = SynthesizerTrn(
            self.cfg.preprocess.n_fft // 2 + 1,
            self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size,
            self.cfg,
        )
        self.model = net_g
        return net_g

    def build_save_dir(self, dataset, speaker):
        save_dir = os.path.join(
            self.args.output_dir,
            "svc_am_step-{}_{}".format(self.am_restore_step, self.args.mode),
        )
        if dataset is not None:
            save_dir = os.path.join(save_dir, "data_{}".format(dataset))
        if speaker != -1:
            save_dir = os.path.join(
                save_dir,
                "spk_{}".format(speaker),
            )
        os.makedirs(save_dir, exist_ok=True)
        print("Saving to ", save_dir)
        return save_dir

    def _build_dataloader(self):
        datasets, collate = self._build_test_dataset()
        self.test_dataset = datasets(self.args, self.cfg, self.infer_type)
        self.test_collate = collate(self.cfg)
        self.test_batch_size = min(
            self.cfg.inference.batch_size, len(self.test_dataset.metadata)
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            collate_fn=self.test_collate,
            num_workers=1,
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        return test_dataloader

    @torch.inference_mode()
    def inference(self):
        res = []
        for i, batch in enumerate(self.test_dataloader):
            pred_audio_list = self._inference_each_batch(batch)
            for j, wav in enumerate(pred_audio_list):
                uid = self.test_dataset.metadata[i * self.test_batch_size + j]["Uid"]
                file = os.path.join(self.args.output_dir, f"{uid}.wav")
                print(f"Saving {file}")

                wav = wav.numpy(force=True)
                save_audio(
                    file,
                    wav,
                    self.cfg.preprocess.sample_rate,
                    add_silence=False,
                    turn_up=not is_silence(wav, self.cfg.preprocess.sample_rate),
                )
                res.append(file)
        return res

    def _inference_each_batch(self, batch_data, noise_scale=0.667):
        device = self.accelerator.device
        pred_res = []
        self.model.eval()
        with torch.no_grad():
            # Put the data to device
            # device = self.accelerator.device
            for k, v in batch_data.items():
                batch_data[k] = v.to(device)

            audios, f0 = self.model.infer(batch_data, noise_scale=noise_scale)

            pred_res.extend(audios)

        return pred_res
