# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torchaudio

from utils.io import save_audio
from utils.audio import load_audio_torch


# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    """
    Copy from: https://github.com/openvpi/audio-slicer/blob/main/slicer2.py
    """

    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 10,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        begin = begin * self.hop_size
        if len(waveform.shape) > 1:
            end = min(waveform.shape[1], end * self.hop_size)
            return waveform[:, begin:end], begin, end
        else:
            end = min(waveform.shape[0], end * self.hop_size)
            return waveform[begin:end], begin, end

    # @timeit
    def slice(self, waveform, return_chunks_positions=False):
        if len(waveform.shape) > 1:
            # (#channle, wave_len) -> (wave_len)
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            chunks_pos_of_waveform = []

            if sil_tags[0][0] > 0:
                chunk, begin, end = self._apply_slice(waveform, 0, sil_tags[0][0])
                chunks.append(chunk)
                chunks_pos_of_waveform.append((begin, end))

            for i in range(len(sil_tags) - 1):
                chunk, begin, end = self._apply_slice(
                    waveform, sil_tags[i][1], sil_tags[i + 1][0]
                )
                chunks.append(chunk)
                chunks_pos_of_waveform.append((begin, end))

            if sil_tags[-1][1] < total_frames:
                chunk, begin, end = self._apply_slice(
                    waveform, sil_tags[-1][1], total_frames
                )
                chunks.append(chunk)
                chunks_pos_of_waveform.append((begin, end))

            return (
                chunks
                if not return_chunks_positions
                else (
                    chunks,
                    chunks_pos_of_waveform,
                )
            )


def split_utterances_from_audio(
    wav_file,
    output_dir,
    max_duration_of_utterance=10.0,
    min_interval=300,
    db_threshold=-40,
):
    """
    Split a long audio into utterances accoring to the silence (VAD).

    max_duration_of_utterance (second):
        The maximum duration of every utterance (seconds)
    min_interval (millisecond):
        The smaller min_interval is, the more sliced audio clips this script is likely to generate.
    """
    print("File:", wav_file.split("/")[-1])
    waveform, fs = torchaudio.load(wav_file)

    slicer = Slicer(sr=fs, min_interval=min_interval, threshold=db_threshold)
    chunks, positions = slicer.slice(waveform, return_chunks_positions=True)

    durations = [(end - begin) / fs for begin, end in positions]
    print(
        "Slicer's min silence part is {}ms, min and max duration of sliced utterances is {}s and {}s".format(
            min_interval, min(durations), max(durations)
        )
    )

    res_chunks, res_positions = [], []
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) == 1:
            chunk = chunk[None, :]

        begin, end = positions[i]
        assert end - begin == chunk.shape[-1]

        max_wav_len = max_duration_of_utterance * fs
        if chunk.shape[-1] <= max_wav_len:
            res_chunks.append(chunk)
            res_positions.append(positions[i])
        else:
            # TODO: to reserve overlapping and conduct fade-in, fade-out

            # Get segments number
            number = 2
            while chunk.shape[-1] // number >= max_wav_len:
                number += 1
            seg_len = chunk.shape[-1] // number

            # Split
            for num in range(number):
                s = seg_len * num
                t = min(s + seg_len, chunk.shape[-1])

                seg_begin = begin + s
                seg_end = begin + t

                res_chunks.append(chunk[:, s:t])
                res_positions.append((seg_begin, seg_end))

    # Save utterances
    os.makedirs(output_dir, exist_ok=True)
    res = {"fs": int(fs)}
    for i, chunk in enumerate(res_chunks):
        filename = "{:04d}.wav".format(i)
        res[filename] = [int(p) for p in res_positions[i]]
        save_audio(os.path.join(output_dir, filename), chunk, fs)

    # Save positions
    with open(os.path.join(output_dir, "positions.json"), "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    return res


def is_silence(
    wavform,
    fs,
    threshold=-40.0,
    min_interval=300,
    hop_size=10,
    min_length=5000,
):
    """
    Detect whether the given wavform is a silence

    wavform: (T, )
    """
    threshold = 10 ** (threshold / 20.0)

    hop_size = round(fs * hop_size / 1000)
    win_size = min(round(min_interval), 4 * hop_size)
    min_length = round(fs * min_length / 1000 / hop_size)

    if wavform.shape[0] <= min_length:
        return True

    # (#Frame,)
    rms_array = get_rms(y=wavform, frame_length=win_size, hop_length=hop_size).squeeze(
        0
    )
    return (rms_array < threshold).all()


def split_audio(
    wav_file, target_sr, output_dir, max_duration_of_segment=10.0, overlap_duration=1.0
):
    """
    Split a long audio into segments.

    target_sr:
        The target sampling rate to save the segments.
    max_duration_of_utterance (second):
        The maximum duration of every utterance (second)
    overlap_duraion:
        Each segment has "overlap duration" (second) overlap with its previous and next segment
    """
    # (#channel, T) -> (T,)
    waveform, fs = torchaudio.load(wav_file)
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=fs, new_freq=target_sr
    )
    waveform = torch.mean(waveform, dim=0)

    # waveform, _ = load_audio_torch(wav_file, target_sr)
    assert len(waveform.shape) == 1

    assert overlap_duration < max_duration_of_segment
    length = int(max_duration_of_segment * target_sr)
    stride = int((max_duration_of_segment - overlap_duration) * target_sr)
    chunks = []
    for i in range(0, len(waveform), stride):
        # (length,)
        chunks.append(waveform[i : i + length])
        if i + length >= len(waveform):
            break

    # Save segments
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, chunk in enumerate(chunks):
        uid = "{:04d}".format(i)
        filename = os.path.join(output_dir, "{}.wav".format(uid))
        results.append(
            {"Uid": uid, "Path": filename, "Duration": len(chunk) / target_sr}
        )
        save_audio(
            filename,
            chunk,
            target_sr,
            turn_up=not is_silence(chunk, target_sr),
            add_silence=False,
        )

    return results


def merge_segments_torchaudio(wav_files, fs, output_path, overlap_duration=1.0):
    """Merge the given wav_files (may have overlaps) into a long audio

    fs:
        The sampling rate of the wav files.
    output_path:
        The output path to save the merged audio.
    overlap_duration (float, optional):
        Each segment has "overlap duration" (second) overlap with its previous and next segment. Defaults to 1.0.
    """

    waveforms = []
    for file in wav_files:
        # (T,)
        waveform, _ = load_audio_torch(file, fs)
        waveforms.append(waveform)

    if len(waveforms) == 1:
        save_audio(output_path, waveforms[0], fs, add_silence=False, turn_up=False)
        return

    overlap_len = int(overlap_duration * fs)
    fade_out = torchaudio.transforms.Fade(fade_out_len=overlap_len)
    fade_in = torchaudio.transforms.Fade(fade_in_len=overlap_len)
    fade_in_and_out = torchaudio.transforms.Fade(fade_out_len=overlap_len)

    segments_lens = [len(wav) for wav in waveforms]
    merged_waveform_len = sum(segments_lens) - overlap_len * (len(waveforms) - 1)
    merged_waveform = torch.zeros(merged_waveform_len)

    start = 0
    for index, wav in enumerate(
        tqdm(waveforms, desc="Merge for {}".format(output_path))
    ):
        wav_len = len(wav)

        if index == 0:
            wav = fade_out(wav)
        elif index == len(waveforms) - 1:
            wav = fade_in(wav)
        else:
            wav = fade_in_and_out(wav)

        merged_waveform[start : start + wav_len] = wav
        start += wav_len - overlap_len

    save_audio(output_path, merged_waveform, fs, add_silence=False, turn_up=True)


def merge_segments_encodec(wav_files, fs, output_path, overlap_duration=1.0):
    """Merge the given wav_files (may have overlaps) into a long audio

    fs:
        The sampling rate of the wav files.
    output_path:
        The output path to save the merged audio.
    overlap_duration (float, optional):
        Each segment has "overlap duration" (second) overlap with its previous and next segment. Defaults to 1.0.
    """

    waveforms = []
    for file in wav_files:
        # (T,)
        waveform, _ = load_audio_torch(file, fs)
        waveforms.append(waveform)

    if len(waveforms) == 1:
        save_audio(output_path, waveforms[0], fs, add_silence=False, turn_up=False)
        return

    device = waveforms[0].device
    dtype = waveforms[0].dtype
    shape = waveforms[0].shape[:-1]

    overlap_len = int(overlap_duration * fs)
    segments_lens = [len(wav) for wav in waveforms]
    merged_waveform_len = sum(segments_lens) - overlap_len * (len(waveforms) - 1)

    sum_weight = torch.zeros(merged_waveform_len, device=device, dtype=dtype)
    out = torch.zeros(*shape, merged_waveform_len, device=device, dtype=dtype)
    offset = 0

    for frame in waveforms:
        frame_length = frame.size(-1)
        t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=torch.float32)[
            1:-1
        ]
        weight = 0.5 - (t - 0.5).abs()
        weighted_frame = frame * weight

        cur = out[..., offset : offset + frame_length]
        cur += weighted_frame[..., : cur.size(-1)]
        out[..., offset : offset + frame_length] = cur

        cur = sum_weight[offset : offset + frame_length]
        cur += weight[..., : cur.size(-1)]
        sum_weight[offset : offset + frame_length] = cur

        offset += frame_length - overlap_len

    assert sum_weight.min() > 0
    merged_waveform = out / sum_weight
    save_audio(output_path, merged_waveform, fs, add_silence=False, turn_up=True)
