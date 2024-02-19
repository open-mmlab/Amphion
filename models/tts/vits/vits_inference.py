# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import numpy as np
from tqdm import tqdm
import torch
import json
from models.tts.base.tts_inferece import TTSInference
from models.tts.vits.vits_dataset import VITSTestDataset, VITSTestCollator
from models.tts.vits.vits import SynthesizerTrn
from processors.phone_extractor import phoneExtractor
from text.text_token_collation import phoneIDCollation
from utils.data_utils import *


class VitsInference(TTSInference):
    def __init__(self, args=None, cfg=None):
        TTSInference.__init__(self, args, cfg)

    def _build_model(self):
        net_g = SynthesizerTrn(
            self.cfg.model.text_token_num,
            self.cfg.preprocess.n_fft // 2 + 1,
            self.cfg.preprocess.segment_size // self.cfg.preprocess.hop_size,
            **self.cfg.model,
        )

        return net_g

    def _build_test_dataset(sefl):
        return VITSTestDataset, VITSTestCollator

    def build_save_dir(self, dataset, speaker):
        save_dir = os.path.join(
            self.args.output_dir,
            "tts_am_step-{}_{}".format(self.am_restore_step, self.args.mode),
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

    def inference_for_batches(
        self, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
    ):
        ###### Construct test_batch ######
        n_batch = len(self.test_dataloader)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(
            "Model eval time: {}, batch_size = {}, n_batch = {}".format(
                now, self.test_batch_size, n_batch
            )
        )
        self.model.eval()

        ###### Inference for each batch ######
        pred_res = []
        with torch.no_grad():
            for i, batch_data in enumerate(
                self.test_dataloader if n_batch == 1 else tqdm(self.test_dataloader)
            ):
                spk_id = None
                if (
                    self.cfg.preprocess.use_spkid
                    and self.cfg.train.multi_speaker_training
                ):
                    spk_id = batch_data["spk_id"]

                outputs = self.model.infer(
                    batch_data["phone_seq"],
                    batch_data["phone_len"],
                    spk_id,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                )

                audios = outputs["y_hat"]
                masks = outputs["mask"]

                for idx in range(audios.size(0)):
                    audio = audios[idx, 0, :].data.cpu().float()
                    mask = masks[idx, :, :]
                    audio_length = (
                        mask.sum([0, 1]).long() * self.cfg.preprocess.hop_size
                    )
                    audio_length = audio_length.cpu().numpy()
                    audio = audio[:audio_length]
                    pred_res.append(audio)

        return pred_res

    def inference_for_single_utterance(
        self, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
    ):
        text = self.args.text

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
        phone_id_seq = phon_id_collator.get_phone_id_sequence(self.cfg, phone_seq)

        if self.cfg.preprocess.add_blank:
            phone_id_seq = intersperse(phone_id_seq, 0)

        # convert phone sequence to phone id sequence
        phone_id_seq = np.array(phone_id_seq)
        phone_id_seq = torch.from_numpy(phone_id_seq)

        # get speaker id if multi-speaker training and use speaker id
        speaker_id = None
        if self.cfg.preprocess.use_spkid and self.cfg.train.multi_speaker_training:
            spk2id_file = os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            with open(spk2id_file, "r") as f:
                spk2id = json.load(f)
                speaker_name = self.args.speaker_name
                assert (
                    speaker_name in spk2id
                ), f"Speaker {speaker_name} not found in the spk2id keys. \
                    Please make sure you've specified the correct speaker name in infer_speaker_name."
                speaker_id = spk2id[speaker_name]
                speaker_id = torch.from_numpy(
                    np.array([speaker_id], dtype=np.int32)
                ).unsqueeze(0)

        with torch.no_grad():
            x_tst = phone_id_seq.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phone_id_seq.size(0)]).to(self.device)
            if speaker_id is not None:
                speaker_id = speaker_id.to(self.device)
            outputs = self.model.infer(
                x_tst,
                x_tst_lengths,
                sid=speaker_id,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )

            audio = outputs["y_hat"][0, 0].data.cpu().float().numpy()

        return audio
