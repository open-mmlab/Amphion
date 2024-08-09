#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from models.tts.UniCATS.CTXvec2wav.datasets import MelSCPDataset
from models.tts.UniCATS.CTXvec2wav.utils import load_model


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained Parallel WaveGAN Generator "
        "(See detail in ctx_vec2wav/bin/decode.py)."
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file. "
        "you need to specify either feats-scp or dumpdir.",
    )
    parser.add_argument(
        "--prompt-scp", 
        default=None, 
        type=str
    )
    parser.add_argument(
        "--xvector-scp", 
        default=None, 
        type=str
    )
    parser.add_argument(
        "--num-frames", 
        default=None, 
        type=str
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        help="directory including feature files. "
        "you need to specify either feats-scp or dumpdir.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )
    parser.add_argument(
        "--normalize-before",
        default=False,
        action="store_true",
        help="whether to perform feature normalization before input to the model. "
        "if true, it assumes that the feature is de-normalized. this is useful when "
        "text2mel model and vocoder use different feature statistics.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.feats_scp is not None and args.dumpdir is not None) or (
        args.feats_scp is None and args.dumpdir is None
    ):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get dataset
    dataset = MelSCPDataset(
        vqidx_scp=args.feats_scp,
        prompt_scp=args.prompt_scp,
        utt2num_frames=args.num_frames,
        return_utt_id=True,
    )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using GPU.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")
    model = load_model(args.checkpoint, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    if args.normalize_before:
        assert hasattr(model, "mean"), "Feature stats are not registered."
        assert hasattr(model, "scale"), "Feature stats are not registered."
    model.backend.remove_weight_norm()
    model = model.eval().to(device)

    # load vq codebook
    feat_codebook = torch.tensor(np.load(config["vq_codebook"], allow_pickle=True)).to(device)  # (2, 320, 256)
    feat_codebook_numgroups = feat_codebook.shape[0]
    feat_codebook = torch.nn.ModuleList([torch.nn.Embedding.from_pretrained(feat_codebook[i], freeze=True) for i in range(feat_codebook_numgroups)])

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, batch in enumerate(pbar, 1):
            utt_id, c, prompt = batch[0], batch[1], batch[2]

            c = torch.tensor(c).to(device)  # (L, D)
            prompt = torch.tensor(prompt).unsqueeze(0).to(device)  # (1, L', 80)

            # slice vq vector
            if c.size(1) > 2:
                vqidx = c[:, 3:5].long()  # (L, 2)
            else:
                vqidx = c.long()
            vqvec = torch.cat([feat_codebook[i](vqidx[:, i]) for i in range(feat_codebook_numgroups)], dim=-1).unsqueeze(0)  # (1, L, 512)

            # generate
            start = time.time()
            y = model.inference(vqvec, prompt, normalize_before=args.normalize_before)[-1].view(-1)
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(config["outdir"], f"{utt_id}.wav"),
                y.cpu().numpy(),
                config["sampling_rate"],
                "PCM_16",
            )

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()
