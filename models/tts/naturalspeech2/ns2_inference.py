# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
import soundfile as sf
import numpy as np

from models.tts.naturalspeech2.ns2 import NaturalSpeech2
from encodec import EncodecModel
from encodec.utils import convert_audio
from utils.util import load_config

from text import text_to_sequence
from text.cmudict import valid_symbols
from text.g2p import preprocess_english, read_lexicon

import torchaudio


class NS2Inference:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args

        self.model = self.build_model()
        self.codec = self.build_codec()

        self.symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
        self.phone2id = {s: i for i, s in enumerate(self.symbols)}
        self.id2phone = {i: s for s, i in self.phone2id.items()}

    def build_model(self):
        model = NaturalSpeech2(self.cfg.model)
        model.load_state_dict(
            torch.load(
                os.path.join(self.args.checkpoint_path, "pytorch_model.bin"),
                map_location="cpu",
            )
        )
        model = model.to(self.args.device)
        return model

    def build_codec(self):
        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model = encodec_model.to(device=self.args.device)
        encodec_model.set_target_bandwidth(12.0)
        return encodec_model

    def get_ref_code(self):
        ref_wav_path = self.args.ref_audio
        ref_wav, sr = torchaudio.load(ref_wav_path)
        ref_wav = convert_audio(
            ref_wav, sr, self.codec.sample_rate, self.codec.channels
        )
        ref_wav = ref_wav.unsqueeze(0).to(device=self.args.device)

        with torch.no_grad():
            encoded_frames = self.codec.encode(ref_wav)
            ref_code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        # print(ref_code.shape)

        ref_mask = torch.ones(ref_code.shape[0], ref_code.shape[-1]).to(ref_code.device)
        # print(ref_mask.shape)

        return ref_code, ref_mask

    def inference(self):
        ref_code, ref_mask = self.get_ref_code()

        lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)
        phone_seq = preprocess_english(self.args.text, lexicon)
        print(phone_seq)

        phone_id = np.array(
            [
                *map(
                    self.phone2id.get,
                    phone_seq.replace("{", "").replace("}", "").split(),
                )
            ]
        )
        phone_id = torch.from_numpy(phone_id).unsqueeze(0).to(device=self.args.device)
        print(phone_id)

        x0, prior_out = self.model.inference(
            ref_code, phone_id, ref_mask, self.args.inference_step
        )
        print(prior_out["dur_pred"])
        print(prior_out["dur_pred_round"])
        print(torch.sum(prior_out["dur_pred_round"]))

        latent_ref = self.codec.quantizer.vq.decode(ref_code.transpose(0, 1))

        rec_wav = self.codec.decoder(x0)
        # ref_wav = self.codec.decoder(latent_ref)

        os.makedirs(self.args.output_dir, exist_ok=True)

        sf.write(
            "{}/{}.wav".format(
                self.args.output_dir, self.args.text.replace(" ", "_", 100)
            ),
            rec_wav[0, 0].detach().cpu().numpy(),
            samplerate=24000,
        )

    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--ref_audio",
            type=str,
            default="",
            help="Reference audio path",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )
        parser.add_argument(
            "--inference_step",
            type=int,
            default=200,
            help="Total inference steps for the diffusion model",
        )
