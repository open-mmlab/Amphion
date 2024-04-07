import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
from models.sgmse.dereverberation.dereverberation import ScoreModel
from models.sgmse.dereverberation.dereverberation_dataset import Specs
from models.sgmse.dereverberation.dereverberation_Trainer import DereverberationTrainer
import json
from os.path import join
import glob
from torchaudio import load
from soundfile import write
from utils.sgmse_util.other import ensure_dir, pad_spec


class DereverberationInference:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.t_eps = self.cfg.train.t_eps
        self.args = args
        self.test_dir = args.test_dir
        self.target_dir = self.args.output_dir
        self.model = self.build_model()
        self.load_state_dict()

    def build_model(self):
        self.model = ScoreModel(self.cfg.model.sgmse)
        return self.model

    def load_state_dict(self):
        self.checkpoint_path = self.args.checkpoint_path
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.cuda(self.args.local_rank)

    def inference(self):
        sr = 16000
        snr = self.args.snr
        N = self.args.N
        corrector_steps = self.args.corrector_steps
        self.model.eval()
        noisy_dir = join(self.test_dir, "noisy/")
        noisy_files = sorted(glob.glob("{}/*.wav".format(noisy_dir)))
        for noisy_file in tqdm(noisy_files):
            filename = noisy_file.split("/")[-1]

            # Load wav
            y, _ = load(noisy_file)
            T_orig = y.size(1)

            # Normalize
            norm_factor = y.abs().max()
            y = y / norm_factor

            # Prepare DNN input
            spec = Specs(self.cfg, subset="", shuffle_spec=False)
            Y = torch.unsqueeze(spec.spec_transform(spec.stft(sig=y.cuda())), 0)
            Y = pad_spec(Y)

            # Reverse sampling
            sampler = DereverberationTrainer.get_pc_sampler(
                self,
                "reverse_diffusion",
                "ald",
                Y.cuda(),
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
            )
            sample, _ = sampler()

            # Backward transform in time domain
            x_hat = spec.istft(sample.squeeze(), T_orig)

            # Renormalize
            x_hat = x_hat * norm_factor

            # Write enhanced wav file
            write(join(self.target_dir, filename), x_hat.cpu().numpy(), 16000)
