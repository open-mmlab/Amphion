# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Amphion. All Rights Reserved
#
################################################################################
"""
Dataset processor tool
"""
import logging
import random
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import copy
import torch.distributed as dist
import os
from pathlib import Path
import string
import time
import transformers

torchaudio.set_audio_backend("soundfile")
AUDIO_FORMAT_SETS = set(["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"])


def _build_semantic_model(semantic_model, mean_var_path, repcodec_model, repcodec_path):
    """Build the w2v semantic model and load pretrained weights."""
    import safetensors

    semantic_model = semantic_model.eval()
    layer_idx = 15
    output_idx = layer_idx + 2
    stat_mean_var = torch.load(mean_var_path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean
    semantic_std = semantic_std

    if repcodec_model is not None:
        safetensors.torch.load_model(repcodec_model, repcodec_path)
        repcodec_model = repcodec_model.eval()
        # print("semantic mean: ", semantic_mean.cpu(), "semantic std: ", semantic_std.cpu())
    return {
        "model": semantic_model,
        "layer_idx": layer_idx,
        "output_idx": output_idx,
        "mean": semantic_mean,
        "std": semantic_std,
        "repcodec_model": repcodec_model,
    }


def segment_w2v(data, segment_length=5 * 50, mode="train"):
    """segmentation (for training codecs)"""
    print("segment length:", segment_length)
    for sample in data:
        if sample["speech_feat"].shape[0] <= segment_length:
            yield sample
        else:
            st = random.randint(0, sample["speech_feat"].shape[0] - segment_length - 1)
            ed = st + segment_length
            sample["speech_feat"] = sample["speech_feat"][st:ed]
            sample["speech_feat_mask"] = sample["speech_feat_mask"][st:ed]
            yield sample


def segment_speech(data, segment_length=5 * 24000, mode="train"):
    """Segment and pad speech data for training codecs."""
    print("Segment speech length:", segment_length)
    for sample in data:
        speech_length = sample["speech"].shape[-1]

        if speech_length <= segment_length:
            # Pad speech to match the segment length
            pad_width = segment_length - speech_length
            sample["speech"] = torch.nn.functional.pad(
                sample["speech"],
                (0, pad_width),  # Pad at the end
                mode="constant",
                value=0,  # Pad with zeros
            )
            yield sample
        else:
            # Randomly crop the speech segment
            st = random.randint(0, speech_length - segment_length - 1)
            ed = st + segment_length
            sample["speech"] = sample["speech"][..., st:ed]
            yield sample


# def loudness_norm(
#     audio: torch.Tensor, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400
# ) -> torch.Tensor:
#     """
#     Perform loudness normalization (ITU-R BS.1770-4) on audio files.

#     Args:
#         audio: audio data
#         rate: sample rate
#         peak: peak normalize audio to N dB. Defaults to -1.0.
#         loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
#         block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)

#     Returns:
#         loudness normalized audio
#     """
#     audio = audio.numpy()
#     # peak normalize audio to [peak] dB
#     audio = pyln.normalize.peak(audio, peak)

#     # measure the loudness first
#     meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
#     _loudness = meter.integrated_loudness(audio)

#     return pyln.normalize.loudness(audio, _loudness, loudness)


def w2v_feature(data, feature_extractor, mode="train", make_multiple_of=1):
    """
    Args:
        data(Iterable[str]): url or local file list
        w2v_path(str): wav2vec2.0 model path
        keep_speech(bool): whether to keep the speech waveform
    """
    for sample in data:
        if sample["sample_rate"] != 16000:
            resampler = torchaudio.transforms.Resample(sample["sample_rate"], 16000)
            tmp_speech = resampler(sample["speech"])
            feats = feature_extractor(
                tmp_speech, sampling_rate=16000, return_tensors="pt"
            )
        else:
            feats = feature_extractor(
                sample["speech"], sampling_rate=16000, return_tensors="pt"
            )

        if "input_values" in feats:  # hubert extractor
            sample["speech_feat"] = feats["input_values"].squeeze(0).squeeze(0)  # (1,T)
            sample["speech_feat_mask"] = torch.ones(1, 1)
            yield sample
        else:  # wav2vec2.0 extractor
            sample["speech_feat"] = feats["input_features"][0]

            sample["speech_feat_mask"] = feats["attention_mask"][0]
            start_idx = sample["speech_feat"].shape[0] % make_multiple_of
            sample["speech_feat"] = sample["speech_feat"][start_idx:]
            sample["speech_feat_mask"] = sample["speech_feat_mask"][start_idx:]
            yield sample


def gluster_opener(
    data,
    mode="train",
    num_epochs=1,
    manual_dist_sampler=False,
    min_seconds=3.0,
    max_seconds=45.0,
):
    """
    WARNING: should set `manual_dist_sampler=False` if the datalist on each process is already disjoint.
    Set it to True if the datalist is the same across all procs, and you want distributed sampler.
    Skip the data that does not belong to this proc.
    """
    if manual_dist_sampler and dist.is_initialized():
        # assert dist.is_initialized(), "Distributed mode requires initialized process group"
        rank = dist.get_rank()  # Get the current process rank
        world_size = dist.get_world_size()  # Total number of processes
        print(
            f"[Rank {rank}] Initialized with manual_dist_sampler=True. Total processes: {world_size}."
        )
    else:
        rank = 0  # Default to rank 0 when not in distributed mode
        world_size = 1  # Treat as single process

    # Iterate through epochs
    for epoch in range(num_epochs):
        # Iterate through samples in the dataset
        for i, sample in enumerate(data):
            # Distributed sampling: only handle samples that match the current process
            if manual_dist_sampler and (i % world_size != rank):
                print(
                    f"[Rank {rank}] Skipping sample {i} in epoch {epoch} (not assigned to this process)."
                )
                continue

            # Create a new sample dictionary with the necessary modifications
            new_sample = copy.deepcopy(sample["src"])
            new_sample["epoch"] = epoch  # Add epoch information

            if hasattr(new_sample, "duration"):
                if new_sample["duration"] < min_seconds:
                    continue
                if new_sample["duration"] > max_seconds:
                    continue

            # Print debug information about the yielded sample
            # if manual_dist_sampler:
            #     print(f"[Rank {rank}] Yielding sample {i} in epoch {epoch}.")

            # Yield the modified sample
            yield new_sample


def gluster_filter(
    data,
    is_emilia=False,
    max_length=10240,
    min_length=10,
    token_max_length=200,
    token_min_length=3,
    min_output_input_ratio=0.0005,
    max_output_input_ratio=0.3,
    ignore_text=False,  # if False, no filtering 'text_token' entry in sample
    mode="train",
    load_from_tar=True,
    make_multiple_of=1,
):
    """Filter sample according to feature and label length
    Inplace operation.

    Args::
        data: Iterable[{key, wav, label, sample_rate}]
        max_length: drop utterance which is greater than max_length(10ms)
        min_length: drop utterance which is less than min_length(10ms)
        token_max_length: drop utterance which is greater than
            token_max_length, especially when use char unit for
            english modeling
        token_min_length: drop utterance which is
            less than token_max_length
        min_output_input_ratio: minimal ration of
            token_length / feats_length(10ms)
        max_output_input_ratio: maximum ration of
            token_length / feats_length(10ms)

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """

    for sample in data:
        # sample['speech'] = torch.randn(100000)
        new_sample = copy.deepcopy(sample)
        start_time = time.time()
        try:
            if is_emilia:
                new_sample["speech"] = torch.tensor(
                    new_sample["mp3"]["array"], dtype=torch.float32
                ).reshape(1, -1)
                del new_sample["mp3"]["array"]
                new_sample["sample_rate"] = new_sample["mp3"]["sampling_rate"]
                new_sample["duration"] = (
                    len(new_sample["speech"]) / new_sample["sample_rate"]
                )
            elif load_from_tar:
                from .gluster_dataset import load_audio_from_tar

                new_sample["speech"], new_sample["sample_rate"] = load_audio_from_tar(
                    new_sample["wav"]
                )
            else:
                new_sample["speech"], new_sample["sample_rate"] = torchaudio.load(
                    new_sample["wav"]
                )

        except Exception as e:
            raise e
        end_time = time.time()
        if (new_sample["speech"].shape[-1] // new_sample["sample_rate"]) > 45.0:
            print("too long audio, skipped")
            continue
        new_sample["speech"] = new_sample["speech"][
            ..., new_sample["speech"].shape[-1] % make_multiple_of :
        ]
        new_sample["load_audio_time"] = end_time - start_time
        if not ignore_text:
            num_frames = new_sample["speech"].size(1) / new_sample["sample_rate"] * 50
            text_token_ratio = len(new_sample["text_token"]) / num_frames
            if (
                text_token_ratio < min_output_input_ratio
                or text_token_ratio > max_output_input_ratio
            ):
                continue
            if (
                len(new_sample["text_token"]) < token_min_length
                or len(new_sample["text_token"]) > token_max_length
            ):
                continue
        yield new_sample
        continue


def resample(data, resample_rate=22050, min_sample_rate=16000, mode="train"):
    """Resample data.
    Inplace operation.

    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        resample_rate: target resample rate

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert "sample_rate" in sample
        assert "speech" in sample
        sample_rate = sample["sample_rate"]
        waveform = sample["speech"]
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample["sample_rate"] = resample_rate
            sample["speech"] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate
            )(waveform)
        max_val = sample["speech"].abs().max()
        if max_val > 1:
            sample["speech"] /= max_val
        yield sample


def compute_fbank(data, feat_extractor, mode="train"):
    """Extract fbank

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        assert "sample_rate" in sample
        assert "speech" in sample
        assert "utt" in sample
        assert "text_token" in sample
        waveform = sample["speech"]
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        sample["speech_feat"] = mat
        del sample["speech"]
        yield sample


def parse_embedding(data, normalize, mode="train"):
    """Parse utt_embedding/spk_embedding

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        sample["utt_embedding"] = torch.tensor(
            sample["utt_embedding"], dtype=torch.float32
        )
        sample["spk_embedding"] = torch.tensor(
            sample["spk_embedding"], dtype=torch.float32
        )
        if normalize:
            sample["utt_embedding"] = F.normalize(sample["utt_embedding"], dim=0)
            sample["spk_embedding"] = F.normalize(sample["spk_embedding"], dim=0)
        yield sample


def tokenize(
    data,
    get_tokenizer,
    allowed_special="all",
    mode="train",
    prepend_language_token=True,
):
    """Decode text to chars or BPE
    Inplace operation

    Args:
        data: Iterable[{key, wav, txt, sample_rate}]

    Returns:
        Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """

    def is_english_string(s):
        # Define the set of allowed characters: all English letters and punctuation
        allowed_chars = set(
            string.ascii_letters + string.punctuation + string.whitespace
        )

        # Check if all characters in the string are within the allowed set
        return all(char in allowed_chars for char in s)

    tokenizer = get_tokenizer()
    for sample in data:
        assert "text" in sample
        sample["text"] = sample["text"].strip()
        sample["text_token"] = tokenizer.encode(
            sample["text"], allowed_special=allowed_special
        )
        if sample["language"] != "en" and is_english_string(sample["text"]):
            continue
        if prepend_language_token:
            sample["text_token"] = [
                tokenizer.to_language_token(sample["language"])
            ] + sample["text_token"]
        yield sample


def shuffle(data, shuffle_size=10000, mode="train"):
    """Local shuffle the data

    Args:
        data: Iterable[{key, feat, label}]
        shuffle_size: buffer size for shuffle

    Returns:
        Iterable[{key, feat, label}]
    """
    buf = []
    if dist.is_initialized():
        rank = dist.get_rank()
        print("RANK {} sort init".format(rank))
    else:
        rank = 0
    # shuffle_size += int(rank) * 1000

    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, ignore_text=True, mode="train"):
    """Sort the data by feature length.
    Sort is used after shuffle and before batch, so we can group
    utts with similar lengths into a batch, and `sort_size` should
    be less than `shuffle_size`

    Args:
        data: Iterable[{key, feat, label}]
        sort_size: buffer size for sort
        ignore text: not calculate text token when sorting

    Returns:
        Iterable[{key, feat, label}]
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        print("RANK {} sort init".format(rank))
    else:
        rank = 0
    buf = []

    if dist.is_initialized():
        rank = dist.get_rank()
        print("RANK {} sort init".format(rank))
    else:
        rank = 0
    # sort_size += int(rank) * 500

    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            if not ignore_text:
                buf.sort(
                    key=lambda x: x["duration"] + len(x["text_token"]),
                    reverse=bool(rank % 2),
                )
            else:
                buf.sort(key=lambda x: x["duration"], reverse=bool(rank % 2))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x["duration"])
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """Static batch the data by `batch_size`

    Args:
        data: Iterable[{key, feat, label}]
        batch_size: batch size

    Returns:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(
    data,
    max_frames_in_batch=12000,
    max_batch_size=50,
    mode="train",
    ignore_text=False,
    min_factor=0.95,
    max_factor=1.5,
):
    """Dynamic batch data with a quadratic exponent that scales based on sequence length."""
    buf = []
    longest_frames = 0

    for sample in data:
        new_sample_frames = sample["speech_feat"].shape[0]
        if not ignore_text:
            new_sample_frames += len(sample["text_token"])

        # Dynamically adjust the quadratic factor based on sequence length
        length_ratio = new_sample_frames / max_frames_in_batch
        quadratic_factor = (
            min_factor + (max_factor - min_factor) * length_ratio
        )  # Scales within [min_factor, max_factor]

        # Apply the dynamic quadratic factor to `new_sample_frames`
        adjusted_frames = int(new_sample_frames**quadratic_factor)
        longest_frames = max(longest_frames, adjusted_frames)

        frames_after_padding = longest_frames * (len(buf) + 1)

        # Check batch size and frame constraints
        if frames_after_padding > max_frames_in_batch or len(buf) >= max_batch_size:
            if buf == []:
                yield [sample]
                longest_frames = 0
            else:
                yield buf
                buf = [sample]
                longest_frames = adjusted_frames
        else:
            buf.append(sample)

    if len(buf) > 0:
        yield buf


def batch(
    data,
    batch_type="static",
    batch_size=16,
    max_frames_in_batch=12000,
    mode="train",
    ignore_text=False,
):
    """Wrapper for static/dynamic batch"""
    if mode == "inference":
        return static_batch(data, 1)
    else:
        if batch_type == "static":
            return static_batch(data, batch_size)
        elif batch_type == "dynamic":
            return dynamic_batch(data, max_frames_in_batch, ignore_text=ignore_text)
        else:
            logging.fatal("Unsupported batch type {}".format(batch_type))


def gluster_padding(
    data,
    use_spk_embedding=False,
    ignore_text=False,
    return_speech=False,
    extract_spec=False,
    mode="train",
):
    """Padding the data into training data

    Args:
        data: Iterable[List[{key, feat, label}]]

    Returns:
        Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        packed_batch_features = {}
        try:
            speech_feat = [i["speech_feat"] for i in sample]
            speech_feat = pad_sequence(speech_feat, batch_first=True, padding_value=0)
            packed_batch_features["input_features"] = (
                speech_feat.contiguous()
            )  # w2v features

            packed_batch_features["attention_mask"] = pad_sequence(
                [utt["speech_feat_mask"].float() for utt in sample], batch_first=True
            ).contiguous()

            packed_batch_features["speech_token_len"] = torch.tensor(
                [len(utt["speech_feat_mask"]) for utt in sample]
            ).contiguous()
        except Exception as e:
            print(e)

        if not ignore_text:
            text_token = [torch.tensor(i["text_token"]) for i in sample]
            text_token_len = torch.tensor(
                [i.size(0) for i in text_token], dtype=torch.int32
            )
            text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
            packed_batch_features["text_token"] = text_token.contiguous()
            packed_batch_features["text_token_len"] = text_token_len.contiguous()

        if return_speech:
            packed_batch_features["speech"] = pad_sequence(
                [utt["speech"].squeeze(0) for utt in sample], batch_first=True
            ).contiguous()
            packed_batch_features["speech_lens"] = torch.tensor(
                [utt["speech"].squeeze(0).__len__() for utt in sample]
            ).contiguous()

        if extract_spec:
            import s3tokenizer

            mels = []
            for b in sample:
                # s3tokenizer uses 16khz mel
                if b["sample_rate"] != 16000:
                    b["speech"] = torchaudio.functional.resample(
                        b["speech"], b["sample_rate"], 16000
                    )

                audio = b["speech"][0]  # get the first channel
                mels.append(s3tokenizer.log_mel_spectrogram(audio))
            packed_batch_features["mels"], packed_batch_features["mels_lens"] = (
                s3tokenizer.padding(mels)
            )

        if not use_spk_embedding:
            packed_batch_features["embedding"] = None  # no speaker embedding

        for key in packed_batch_features.keys():
            if isinstance(packed_batch_features[key], torch.Tensor):
                if torch.isnan(packed_batch_features[key]).any():
                    print("NaN found in preprocessor, key", key)
                    continue
        packed_batch_features["epoch"] = sample[0]["epoch"]
        packed_batch_features["duration"] = sum(s["duration"] for s in sample)
        packed_batch_features["load_audio_time"] = max(
            s["load_audio_time"] for s in sample
        )
        packed_batch_features["sample_rate"] = sample[0]["sample_rate"]
        yield packed_batch_features
