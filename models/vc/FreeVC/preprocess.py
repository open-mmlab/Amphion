from models.vc.FreeVC.wavlm import load_wavlm
from models.vc.FreeVC.hifigan import load_hifigan
from models.vc.FreeVC.mel_processing import mel_spectrogram_torch
from utils.util import load_config

from speaker_encoder.voice_encoder import SpeakerEncoder
import speaker_encoder.audio

import os
import random
from typing import Optional
import argparse

import torch
import numpy as np
import librosa
from scipy.io import wavfile
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from glob import glob
import torchaudio
import torchvision.transforms.v2


def downsample(args):
    in_dir, wav_name, target = args

    speaker = wav_name[:4]
    wav_path = os.path.join(in_dir, speaker, wav_name)

    if not os.path.exists(wav_path):
        return

    # speaker 's5', 'p280', 'p315' are excluded,
    if "_mic2.flac" not in wav_path:
        return

    wav = None

    for out_dir, target_sr in target:
        save_name = wav_name.replace("_mic2.flac", ".wav")
        save_path = os.path.join(out_dir, speaker, save_name)
        if os.path.exists(save_path):
            continue

        if wav is None:
            wav, src_sr = librosa.load(wav_path)
            wav, _ = librosa.effects.trim(wav, top_db=20)
            peak = np.abs(wav).max()
            if peak > 1.0:
                wav = 0.98 * wav / peak

        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        target_wav = librosa.resample(wav, orig_sr=src_sr, target_sr=target_sr)
        wavfile.write(
            save_path, target_sr, (target_wav * np.iinfo(np.int16).max).astype(np.int16)
        )


def resample_vctk(*, vctk_dir, vctk_16k_dir, vctk_22k_dir):
    print("Start resampling VCTK dataset...")

    target = [(vctk_16k_dir, 16000), (vctk_22k_dir, 22050)]

    pool = Pool(processes=cpu_count() - 2)

    in_dir = os.path.join(vctk_dir, "wav48_silence_trimmed")

    wav_names = []
    for speaker in os.listdir(in_dir):
        spk_dir = os.path.join(vctk_dir, speaker)
        if os.path.isdir(spk_dir):
            wav_names.extend(os.listdir(spk_dir))

    tasks = [(vctk_dir, wav_name, target) for wav_name in wav_names]

    with tqdm(total=len(tasks)) as pbar:
        for _ in pool.imap_unordered(downsample, tasks):
            pbar.update()

    print("Done!")


def generate_split(*, vctk_16k_dir, split_dir):
    print("Start generating split...")

    src_dir = vctk_16k_dir

    train = []
    val = []
    test = []

    for speaker in os.listdir(src_dir):
        wav_names = os.listdir(os.path.join(src_dir, speaker))
        random.shuffle(wav_names)
        train.extend(wav_names[2:-10])
        val.extend(wav_names[:2])
        test.extend(wav_names[-10:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    train_list = os.path.join(split_dir, "train.txt")
    val_list = os.path.join(split_dir, "val.txt")
    test_list = os.path.join(split_dir, "test.txt")

    os.makedirs(split_dir, exist_ok=True)

    for list_path, wav_names in zip(
        [train_list, val_list, test_list], [train, val, test]
    ):
        with open(list_path, "w") as f:
            for wav_name in wav_names:
                speaker = wav_name[:4]
                f.write(f"{speaker}/{wav_name}" + "\n")

    print("Done!")


def preprocess_spk(*, vctk_16k_dir, preprocess_spk_dir):
    in_dir = vctk_16k_dir
    out_dir = preprocess_spk_dir

    wav_names = []
    for speaker in os.listdir(in_dir):
        spk_dir = os.path.join(in_dir, speaker)
        if os.path.isdir(spk_dir):
            wav_names.extend(os.listdir(spk_dir))

    pretrained_spk_ckpt_path = os.path.join(
        os.path.dirname(__file__), "speaker_encoder/ckpt/pretrained_bak_5805000.pt"
    )
    spk_encoder = SpeakerEncoder(pretrained_spk_ckpt_path)

    for wav_name in tqdm(wav_names):
        speaker = wav_name[:4]
        save_path = os.path.join(out_dir, speaker, wav_name.replace(".wav", ".pt"))

        if os.path.exists(save_path):
            continue

        wav_path = os.path.join(in_dir, speaker, wav_name)
        spk_wav = speaker_encoder.audio.preprocess_wav(wav_path)
        spk = spk_encoder.embed_utterance(spk_wav)
        spk = torch.from_numpy(spk)

        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        torch.save(spk, save_path)


@torch.no_grad()
def calc_ssl_features(wavlm, wav):
    return wavlm(wav).last_hidden_state.transpose(1, 2)


def preprocess_ssl(*, vctk_16k_dir, preprocess_ssl_dir):
    print("Start preprocessing SSL features...")

    in_dir = vctk_16k_dir
    out_dir = preprocess_ssl_dir
    sr = 16000

    model = load_wavlm().cuda()  # type:ignore
    filenames = glob(f"{in_dir}/*/*.wav", recursive=True)

    for filename in tqdm(filenames):
        wav_name = os.path.basename(filename)
        speaker = wav_name[:4]

        save_dir = os.path.join(out_dir, speaker)
        save_path = os.path.join(save_dir, wav_name.replace(".wav", ".pt"))
        if os.path.exists(save_path):
            continue

        os.makedirs(save_dir, exist_ok=True)
        wav, _ = librosa.load(filename, sr=sr)
        wav = torch.from_numpy(wav).unsqueeze_(0).cuda()
        ssl_features = calc_ssl_features(model, wav)
        torch.save(ssl_features.cpu(), save_path)

    print("Done!")


def mel_resize(mel, height):  # 68-92
    tgt = torchvision.transforms.v2.functional.resize(mel, [height, mel.size(-1)])
    if height >= mel.size(-2):
        return tgt[:, : mel.size(-2), :]
    else:
        silence = tgt[:, -1:, :].repeat(1, mel.size(-2) - height, 1)
        silence += torch.randn_like(silence) / 10
        return torch.cat((tgt, silence), 1)


@torch.no_grad()
def preprocess_sr(
    *,
    vctk_22k_dir: str,
    preprocess_sr_dir: str,
    hifigan_ckpt_path: str,
    minh: int = 68,
    maxh: int = 92,
    cuda_rank: Optional[int] = None,
    cuda_total: Optional[int] = None,
):
    assert 68 <= minh <= maxh <= 92

    in_dir = vctk_22k_dir
    out_dir = preprocess_sr_dir

    wavlm = load_wavlm()
    hifigan, hifigan_config = load_hifigan(hifigan_ckpt_path)

    device = (
        torch.device(f"cuda:{cuda_rank}")
        if cuda_rank is not None
        else torch.device("cuda")
    )

    wavlm = wavlm.to(device)  # type:ignore
    hifigan = hifigan.to(device)  # type:ignore

    target_sr = 16000
    resample = torchaudio.transforms.Resample(
        orig_freq=hifigan_config.sampling_rate, new_freq=target_sr
    ).to(device)

    filenames = glob(f"{in_dir}/*/*.wav", recursive=True)
    filenames.sort()

    if cuda_rank is not None:
        assert cuda_total is not None
        filenames = filenames[cuda_rank::cuda_total]

    with tqdm(total=len(filenames) * (maxh - minh + 1)) as pbar:
        for filename in filenames:
            wav_name = os.path.basename(filename)
            speaker = wav_name[:4]

            odir = os.path.join(out_dir, speaker)
            os.makedirs(odir, exist_ok=True)

            wav, sr = torchaudio.load(filename)
            assert sr == hifigan_config.sampling_rate
            wav = wav.to(device)

            mel = mel_spectrogram_torch(
                wav,
                n_fft=hifigan_config.n_fft,
                num_mels=hifigan_config.num_mels,
                sampling_rate=hifigan_config.sampling_rate,
                hop_size=hifigan_config.hop_size,
                win_size=hifigan_config.win_size,
                fmin=hifigan_config.fmin,
                fmax=hifigan_config.fmax,
            )

            for h in range(minh, maxh + 1):
                ssl_path = os.path.join(odir, wav_name.replace(".wav", f"_{h}.pt"))
                wav_path = os.path.join(odir, wav_name.replace(".wav", f"_{h}.wav"))

                if not os.path.exists(wav_path):
                    mel_rs = mel_resize(mel, h)

                    wav_rs = hifigan(mel_rs)[0]
                    assert wav_rs.shape[0] == 1

                    wav_rs = resample(wav_rs)

                    ssl_features = calc_ssl_features(wavlm, wav_rs)
                    torch.save(ssl_features.cpu(), ssl_path)
                    wavfile.write(wav_path, target_sr, wav_rs.cpu().numpy().squeeze(0))

                pbar.update()


def preprocess(cfg, args):
    resample_vctk(
        vctk_dir=cfg.preprocess.vctk_dir,
        vctk_16k_dir=cfg.preprocess.vctk_16k_dir,
        vctk_22k_dir=cfg.preprocess.vctk_22k_dir,
    )
    generate_split(
        vctk_16k_dir=cfg.preprocess.vctk_16k_dir,
        split_dir=cfg.preprocess.split_dir,
    )
    preprocess_spk(
        vctk_16k_dir=cfg.preprocess.vctk_16k_dir,
        preprocess_spk_dir=cfg.preprocess.spk_dir,
    )
    preprocess_ssl(
        vctk_16k_dir=cfg.preprocess.vctk_16k_dir,
        preprocess_ssl_dir=cfg.preprocess.ssl_dir,
    )
    preprocess_sr(
        vctk_22k_dir=cfg.preprocess.vctk_22k_dir,
        preprocess_sr_dir=cfg.preprocess.sr_dir,
        hifigan_ckpt_path=cfg.preprocess.hifigan_ckpt_path,
        minh=cfg.preprocess.minh,
        maxh=cfg.preprocess.maxh,
        cuda_rank=args.cuda_rank,
        cuda_total=args.cuda_total,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--cuda_rank", type=int, default=None)
    parser.add_argument("--cuda_total", type=int, default=None)

    args = parser.parse_args()
    cfg = load_config(args.config)

    preprocess(cfg, args)


if __name__ == "__main__":
    main()
