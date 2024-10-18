# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from models.tts.maskgct.g2p.g2p import PhonemeBpeTokenizer
from models.tts.maskgct.g2p.utils.g2p import phonemizer_g2p
import tqdm
from typing import List
import json
import os
import re


def ph_g2p(text, language):

    return phonemizer_g2p(text=text, language=language)


def g2p(text, sentence, language):

    return text_tokenizer.tokenize(text=text, sentence=sentence, language=language)


def is_chinese(char):
    if char >= "\u4e00" and char <= "\u9fa5":
        return True
    else:
        return False


def is_alphabet(char):
    if (char >= "\u0041" and char <= "\u005a") or (
        char >= "\u0061" and char <= "\u007a"
    ):
        return True
    else:
        return False


def is_other(char):
    if not (is_chinese(char) or is_alphabet(char)):
        return True
    else:
        return False


def get_segment(text: str) -> List[str]:
    # sentence --> [ch_part, en_part, ch_part, ...]
    segments = []
    types = []
    flag = 0
    temp_seg = ""
    temp_lang = ""

    # Determine the type of each character. type: blank, chinese, alphabet, number, unk and point.
    for i, ch in enumerate(text):
        if is_chinese(ch):
            types.append("zh")
        elif is_alphabet(ch):
            types.append("en")
        else:
            types.append("other")

    assert len(types) == len(text)

    for i in range(len(types)):
        # find the first char of the seg
        if flag == 0:
            temp_seg += text[i]
            temp_lang = types[i]
            flag = 1
        else:
            if temp_lang == "other":
                if types[i] == temp_lang:
                    temp_seg += text[i]
                else:
                    temp_seg += text[i]
                    temp_lang = types[i]
            else:
                if types[i] == temp_lang:
                    temp_seg += text[i]
                elif types[i] == "other":
                    temp_seg += text[i]
                else:
                    segments.append((temp_seg, temp_lang))
                    temp_seg = text[i]
                    temp_lang = types[i]
                    flag = 1

    segments.append((temp_seg, temp_lang))
    return segments


def chn_eng_g2p(text: str):
    # now only en and ch
    segments = get_segment(text)
    all_phoneme = ""
    all_tokens = []

    for index in range(len(segments)):
        seg = segments[index]
        phoneme, token = g2p(seg[0], text, seg[1])
        all_phoneme += phoneme + "|"
        all_tokens += token

        if seg[1] == "en" and index == len(segments) - 1 and all_phoneme[-2] == "_":
            all_phoneme = all_phoneme[:-2]
            all_tokens = all_tokens[:-1]
    return all_phoneme, all_tokens


text_tokenizer = PhonemeBpeTokenizer()
with open("./models/tts/maskgct/g2p/g2p/vocab.json", "r") as f:
    json_data = f.read()
data = json.loads(json_data)
vocab = data["vocab"]
