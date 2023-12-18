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
from processors.phone_extractor import phoneExtractor
from text.text_token_collation import phoneIDCollation
import numpy as np
import json


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
                self.cfg.preprocess.sample_rate,
                add_silence=True,
                turn_up=True,
            )
            os.remove(os.path.join(self.args.output_dir, f"{uid}.pt"))

    @torch.inference_mode()
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

    def inference_for_single_utterance(self):
        text = self.args.text
        control_values = (
            self.args.pitch_control,
            self.args.energy_control,
            self.args.duration_control,
        )
        pitch_control, energy_control, duration_control = control_values

        # get phone symbol file
        phone_symbol_file = None
        if self.cfg.preprocess.phone_extractor != "lexicon":
            phone_symbol_file = os.path.join(
                self.exp_dir, self.cfg.preprocess.symbols_dict
            )
            assert os.path.exists(phone_symbol_file)
        # convert text to phone sequence
        phone_extractor = phoneExtractor(self.cfg)

        phone_seq = phone_extractor.extract_phone(text)  # phone_seq: list
        # convert phone sequence to phone id sequence
        phon_id_collator = phoneIDCollation(
            self.cfg, symbols_dict_file=phone_symbol_file
        )
        phone_seq = ["{"] + phone_seq + ["}"]
        phone_id_seq = phon_id_collator.get_phone_id_sequence(self.cfg, phone_seq)

        # convert phone sequence to phone id sequence
        phone_id_seq = np.array(phone_id_seq)
        phone_id_seq = torch.from_numpy(phone_id_seq)

        # get speaker id if multi-speaker training and use speaker id
        speaker_id = None
        if self.cfg.preprocess.use_spkid and self.cfg.train.multi_speaker_training:
            spk2id_file = os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            with open(spk2id_file, "r") as f:
                spk2id = json.load(f)
                speaker_id = spk2id[self.args.speaker_name]
                speaker_id = torch.from_numpy(np.array([speaker_id], dtype=np.int32))
        else:
            speaker_id = torch.Tensor(0).view(-1)

        with torch.no_grad():
            x_tst = phone_id_seq.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phone_id_seq.size(0)]).to(self.device)
            if speaker_id is not None:
                speaker_id = speaker_id.to(self.device)

            data = {}
            data["texts"] = x_tst
            data["text_len"] = x_tst_lengths
            data["spk_id"] = speaker_id

            output = self.model(
                data,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )
            pred_res = output["postnet_output"]
            vocoder_cfg, vocoder_ckpt = self._parse_vocoder(self.args.vocoder_dir)
            audio = synthesis(
                cfg=vocoder_cfg,
                vocoder_weight_file=vocoder_ckpt,
                n_samples=None,
                pred=pred_res,
            )
        return audio[0]
