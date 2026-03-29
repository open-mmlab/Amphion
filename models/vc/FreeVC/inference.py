from models.vc.FreeVC.model import SynthesizerTrn
from models.vc.FreeVC.wavlm import load_wavlm
from models.vc.FreeVC.mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder
from utils.util import load_config
from models.vc.FreeVC.train_utils import load_checkpoint
from models.vc.FreeVC.preprocess import calc_ssl_features

import os
import argparse
import torch
import librosa
from scipy.io import wavfile
from tqdm import tqdm
from typing import Any

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str, help="path to pth file")
    parser.add_argument("--convert", type=str, help="path to txt file")
    parser.add_argument("--outdir", type=str, help="path to output dir")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg: Any = load_config(args.config)

    print("Loading model...")
    net_g = SynthesizerTrn(
        cfg.data.filter_length // 2 + 1,  # type:ignore
        cfg.train.segment_size // cfg.data.hop_length,  # type:ignore
        **cfg.model,  # type:ignore
    ).cuda()
    net_g.eval()
    print("Loading checkpoint...")
    _ = load_checkpoint(args.ckpt, net_g, None, True)

    print("Loading WavLM for content...")
    wavlm = load_wavlm().cuda()  # type:ignore

    if cfg.model.use_spk:
        print("Loading speaker encoder...")
        spk_path = os.path.join(
            os.path.dirname(__file__), "speaker_encoder/ckpt/pretrained_bak_5805000.pt"
        )
        smodel = SpeakerEncoder(spk_path)

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.convert, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=cfg.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if cfg.model.use_spk:
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt,
                    cfg.data.filter_length,
                    cfg.data.n_mel_channels,
                    cfg.data.sampling_rate,
                    cfg.data.hop_length,
                    cfg.data.win_length,
                    cfg.data.mel_fmin,
                    cfg.data.mel_fmax,
                )
            # src
            wav_src, _ = librosa.load(src, sr=cfg.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            c = calc_ssl_features(wavlm, wav_src)

            if cfg.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()

            outpath = os.path.join(args.outdir, f"{title}.wav")
            wavfile.write(outpath, cfg.data.sampling_rate, audio)
