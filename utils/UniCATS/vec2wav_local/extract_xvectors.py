#!/usr/bin/env python3
#  2022, Hitachi LTD.; Nelson Yalta
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import kaldiio
import logging
from pathlib import Path
import sys
import torch
import os
import numpy as np

from tqdm.contrib import tqdm

from espnet2.fileio.sound_scp import SoundScpReader


def get_parser():
    """Construct the parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pretrained_model", type=str, help="Pretrained model.")
    parser.add_argument("--pretrained_model_dir", type=str, help="The directory for saving the pretrained model.")
    parser.add_argument(
        "--toolkit",
        type=str,
        help="Toolkit for Extracting X-vectors.",
        choices=["espnet", "speechbrain"],
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument(
        "in_folder", type=Path, help="Path to the input kaldi data directory."
    )
    parser.add_argument(
        "out_folder",
        type=Path,
        help="Output folder to save the xvectors.",
    )
    return parser


def main(argv):
    """Load the model, generate kernel and bandpass plots."""
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.verbose > 0:
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

    if torch.cuda.is_available() and ("cuda" in args.device):
        device = args.device
    else:
        device = "cpu"

    if args.toolkit == "speechbrain":
        from speechbrain.dataio.preprocess import AudioNormalizer
        from speechbrain.pretrained import EncoderClassifier

        # Prepare spk2utt for mean x-vector
        spk2utt = dict()
        with open(os.path.join(args.in_folder, "spk2utt"), "r") as reader:
            for line in reader:
                details = line.split()
                spk2utt[details[0]] = details[1:]

        # TODO(nelson): The model inference can be moved into functon.
        classifier = EncoderClassifier.from_hparams(
            source=args.pretrained_model, run_opts={"device": device}, savedir=args.pretrained_model_dir
        )
        audio_norm = AudioNormalizer()

        wav_scp = SoundScpReader(os.path.join(args.in_folder, "wav.scp"))
        os.makedirs(args.out_folder, exist_ok=True)
        writer_utt = kaldiio.WriteHelper(
            "ark,scp:{0}/xvector.ark,{0}/xvector.scp".format(args.out_folder)
        )
        writer_spk = kaldiio.WriteHelper(
            "ark,scp:{0}/spk_xvector.ark,{0}/spk_xvector.scp".format(args.out_folder)
        )

        for speaker in tqdm(spk2utt):
            xvectors = list()
            for utt in spk2utt[speaker]:
                in_sr, wav = wav_scp[utt]
                # Amp Normalization -1 ~ 1
                amax = np.amax(np.absolute(wav))
                wav = wav.astype(np.float32) / amax
                # Freq Norm
                wav = audio_norm(torch.from_numpy(wav), in_sr).to(device)
                # X-vector Embedding
                embeds = classifier.encode_batch(wav).detach().cpu().numpy()[0]
                writer_utt[utt] = np.squeeze(embeds)
                xvectors.append(embeds)

            # Speaker Normalization
            embeds = np.mean(np.stack(xvectors, 0), 0)
            writer_spk[speaker] = np.squeeze(embeds)
        writer_utt.close()
        writer_spk.close()

    elif args.toolkit == "espnet":
        raise NotImplementedError(
            "Follow details at: https://github.com/espnet/espnet/issues/3040"
        )
    else:
        raise ValueError(
            f"Unkown type of toolkit. Only supported: speechbrain, espnet, kaldi"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
