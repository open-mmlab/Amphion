# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from tqdm import tqdm
from collections import OrderedDict

from models.tts.base.tts_inferece import TTSInference
from models.tts.fastspeech2.fs2_dataset import FS2TestDataset, FS2TestCollator
from utils.util import load_config
from utils.io import save_audio
from models.tts.fastspeech2.fs2 import FastSpeech2
from models.vocoders.vocoder_inference import synthesis
from pathlib import Path


class FastSpeech2Inference(TTSInference):
    def __init__(self, args, cfg):
        TTSInference.__init__(self, args, cfg)
        self.args = args
        self.cfg = cfg
        self.infer_type = args.mode

    def _build_model(self):
        self.model = FastSpeech2(self.cfg)
        return self.model

    def load_model(self, state_dict):
        raw_dict = state_dict["model"]
        clean_dict = OrderedDict()
        for k, v in raw_dict.items():
            if k.startswith("module."):
                clean_dict[k[7:]] = v
            else:
                clean_dict[k] = v

        self.model.load_state_dict(clean_dict)

    def _build_test_dataset(self):
        return FS2TestDataset, FS2TestCollator

    @staticmethod
    def _parse_vocoder(vocoder_dir):
        r"""Parse vocoder config"""
        vocoder_dir = os.path.abspath(vocoder_dir)
        ckpt_list = [ckpt for ckpt in Path(vocoder_dir).glob("*.pt")]
        # last step (different from the base *int(x.stem)*)
        ckpt_list.sort(
            key=lambda x: int(x.stem.split("_")[-2].split("-")[-1]), reverse=True
        )
        ckpt_path = str(ckpt_list[0])
        vocoder_cfg = load_config(
            os.path.join(vocoder_dir, "args.json"), lowercase=True
        )
        return vocoder_cfg, ckpt_path

    @torch.inference_mode()
    def inference(self):
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
                self.cfg.preprocess.sample_rate,
                add_silence=True,
                turn_up=True,
            )
            os.remove(os.path.join(self.args.output_dir, f"{uid}.pt"))

    def _inference_each_batch(self, batch_data):
        device = self.accelerator.device
        control_values = (
            self.args.pitch_control,
            self.args.energy_control,
            self.args.duration_control,
        )
        for k, v in batch_data.items():
            batch_data[k] = v.to(device)

        pitch_control, energy_control, duration_control = control_values

        output = self.model(
            batch_data,
            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control,
        )
        pred_res = output["postnet_output"]
        mel_lens = output["mel_lens"].cpu()
        return pred_res, mel_lens, 0
