# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import numpy as np
from tqdm import tqdm
import torch

from models.tts.base.tts_inferece import TTSInference
from models.tts.vits.vits_dataset import VITSTestDataset, VITSTestCollator
from models.tts.vits.vits import SynthesizerTrn
from text.symbols import symbols
from text.g2p import preprocess_english, read_lexicon
from text import text_to_sequence


class VitsInference(TTSInference):
    def __init__(self, args=None, cfg=None):
        TTSInference.__init__(self, args, cfg)

    def _build_model(self):
        net_g = SynthesizerTrn(
            len(symbols),
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
                # Put the data to device
                for k, v in batch_data.items():
                    batch_data[k] = batch_data[k]

                outputs = self.model.infer(
                    batch_data["text_seq"],
                    batch_data["text_len"],
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
        self, text, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
    ):
        # convert text to phone sequence
        lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)
        phone_seq = preprocess_english(text, lexicon)

        # convert phone sequence to phone id sequence
        phone_id_seq = text_to_sequence(phone_seq, self.cfg.preprocess.text_cleaners)
        phone_id_seq = np.array(phone_id_seq)
        phone_id_seq = torch.from_numpy(phone_id_seq)

        with torch.no_grad():
            x_tst = phone_id_seq.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phone_id_seq.size(0)]).cuda()
            outputs = self.model.infer(
                x_tst,
                x_tst_lengths,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )

            audio = outputs["y_hat"][0, 0].data.cpu().float().numpy()

        return audio
