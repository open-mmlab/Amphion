# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import tgt


def get_alignment(tier, cfg):
    sample_rate = cfg["sample_rate"]
    hop_size = cfg["hop_size"]

    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0

    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sample_rate / hop_size)
                - np.round(s * sample_rate / hop_size)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def get_duration(utt, wav, cfg):
    speaker = utt["Singer"]
    basename = utt["Uid"]
    dataset = utt["Dataset"]
    sample_rate = cfg["sample_rate"]

    # print(cfg.processed_dir, dataset, speaker, basename)
    wav_path = os.path.join(
        cfg.processed_dir, dataset, "raw_data", speaker, "{}.wav".format(basename)
    )
    text_path = os.path.join(
        cfg.processed_dir, dataset, "raw_data", speaker, "{}.lab".format(basename)
    )
    tg_path = os.path.join(
        cfg.processed_dir, dataset, "TextGrid", speaker, "{}.TextGrid".format(basename)
    )

    # Read raw text
    with open(text_path, "r") as f:
        raw_text = f.readline().strip("\n")

    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones"), cfg
    )
    text = "{" + " ".join(phone) + "}"
    if start >= end:
        return None

    return duration, text, int(sample_rate * start), int(sample_rate * end)
