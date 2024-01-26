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

from models.vc.base import VCInference
from models.vc.vits.vits import SynthesizerTrn

from models.vc.base.vc_dataset import VCTestDataset, VCTestCollator
from utils.io import save_audio
from utils.audio_slicer import is_silence


class VitsInference(VCInference):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        VCInference.__init__(self, args, cfg)

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
            "vc_am_step-{}_{}".format(self.am_restore_step, self.args.mode),
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

    @torch.inference_mode()
    def inference(self):
        res = []
        for i, batch in enumerate(self.test_dataloader):
            pred_audio_list = self._inference_each_batch(batch)
            for it, wav in zip(self.test_dataset.metadata, pred_audio_list):
                uid = it["Uid"]
                file = os.path.join(self.args.output_dir, f"{uid}.wav")

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
