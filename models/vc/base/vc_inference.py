# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch

from models.base.new_inference import BaseInference
from models.vc.base.vc_dataset import VCTestCollator, VCTestDataset

from utils.io import save_audio
from utils.util import load_config
from utils.audio_slicer import is_silence
from models.vocoders.vocoder_inference import synthesis

EPS = 1.0e-12


class VCInference(BaseInference):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        BaseInference.__init__(self, args, cfg, infer_type)

    def _build_test_dataset(self):
        return VCTestDataset, VCTestCollator

    @torch.inference_mode()
    def inference(self):
        for i, batch in enumerate(self.test_dataloader):
            y_pred = self._inference_each_batch(batch).cpu()
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

        vocoder_cfg = load_config(
            os.path.join(self.args.vocoder_dir, "args.json"), lowercase=True
        )

        res = synthesis(
            cfg=vocoder_cfg,
            vocoder_weight_file=self.args.vocoder_dir,
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
