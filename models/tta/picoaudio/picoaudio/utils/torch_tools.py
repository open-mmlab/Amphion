import torch
import torchaudio
import random
import itertools
import numpy as np

# from utils.mix import mix


def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform * 0.5


def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)

    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        pad_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, pad_wav])
        return waveform


def _pad_spec(fbank, target_length=1024):
    batch, n_frames, channels = fbank.shape
    p = target_length - n_frames
    if p > 0:
        pad = torch.zeros(batch, p, channels).to(fbank.device)
        fbank = torch.cat([fbank, pad], 1)
    elif p < 0:
        fbank = fbank[:, :target_length, :]

    if channels % 2 != 0:
        fbank = fbank[:, :, :-1]

    return fbank


def read_wav_file(filename, segment_length):

    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)[0]
    try:
        waveform = normalize_wav(waveform)
    except:
        print("Exception normalizing:", filename)
        waveform = torch.ones(160000)
    waveform = pad_wav(waveform, segment_length).unsqueeze(0)
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    return waveform


def get_mel_from_wav(audio, _stft):
    audio = torch.nan_to_num(torch.clip(audio, -1, 1))
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    return melspec, log_magnitudes_stft, energy


def wav_to_fbank(paths, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    waveform = torch.cat(
        [read_wav_file(path, target_length * 160) for path in paths], 0
    )  # hop size is 160

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform


def uncapitalize(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ""


def mix_wavs_and_captions(path1, path2, caption1, caption2, target_length=1024):
    sound1 = read_wav_file(path1, target_length * 160)[0].numpy()
    sound2 = read_wav_file(path2, target_length * 160)[0].numpy()
    mixed_sound = mix(sound1, sound2, 0.5, 16000).reshape(1, -1)
    mixed_caption = "{} and {}".format(caption1, uncapitalize(caption2))
    return mixed_sound, mixed_caption


def augment(paths, texts, num_items=4, target_length=1024):
    mixed_sounds, mixed_captions = [], []
    combinations = list(itertools.combinations(list(range(len(texts))), 2))
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]

    for i, j in selected_combinations:
        new_sound, new_caption = mix_wavs_and_captions(
            paths[i], paths[j], texts[i], texts[j], target_length
        )
        mixed_sounds.append(new_sound)
        mixed_captions.append(new_caption)

    waveform = torch.tensor(np.concatenate(mixed_sounds, 0))
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform

    return waveform, mixed_captions


def augment_wav_to_fbank(paths, texts, num_items=4, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    waveform, captions = augment(paths, texts)
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform, captions
