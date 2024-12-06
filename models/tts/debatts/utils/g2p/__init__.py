# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#  from https://github.com/keithito/tacotron

import sys
import utils.g2p.cleaners
from tokenizers import Tokenizer
import json
import re


class PhonemeBpeTokenizer:
    def __init__(self, tokenizer_path="./utils/g2p/bpe_613.json"):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer_path = tokenizer_path

        with open(tokenizer_path, "r") as f:
            json_data = f.read()
        data = json.loads(json_data)
        self.vocab = data["model"]["vocab"]

    def tokenize(self, text, language):
        # 1. convert text to phoneme
        phonemes = _clean_text(text, ["cje_cleaners"])
        # print('clean text: ', phonemes)

        # 2. replace blank space " " with "_"
        phonemes = phonemes.replace(" ", "_")

        # 3. tokenize phonemes
        phoneme_tokens = self.tokenizer.encode(phonemes).ids
        # print('encode: ', phoneme_tokens)

        # 4. connect single phoneme because of "`" or "⁼"
        if language == "zh":
            phoneme_tokens = _connect_phone(phoneme_tokens)
            # print('encode phoneme: ', phoneme_tokens)

            # 5. connect tones with previous phoneme
            phoneme_tokens = _connect_tone(phoneme_tokens, self.vocab)
            # print('connect tones: ', phoneme_tokens)

        # 6. decode tokens [optional]
        # decoded_text = self.tokenizer.decode(phoneme_tokens)
        # print('decoded: ', decoded_text)

        # if not len(phoneme_tokens):
        #   raise ValueError("Empty text is given")

        return phonemes, phoneme_tokens


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(utils.g2p.cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)

    return text


def _connect_phone(phoneme_tokens):
    sublist = [
        [32, 66, 67],  # "p⁼wo"
        [32, 66],  # "p⁼"
        [34, 66],  # "t⁼"
        [27, 66],  # "k⁼"
        [78, 66],  # "tʃ⁼"
        [81, 17, 66, 55, 17],  # "ts`⁼ɹ`"
        [81, 17, 66],  # "ts`⁼"
        [81, 17, 61, 55, 17],  # "ts`ʰɹ`"
        [81, 17, 61],  # "ts`ʰ"
        [33, 17, 55, 17],  # "s`ɹ`"
        [33, 17],  # "s`"
        [55, 17, 55, 17],  # "ɹ`ɹ`"
        [55, 17],  # "ɹ`"
        [81, 66, 55],  # "ts⁼ɹ"
        [81, 66],  # "ts⁼"
        [48, 55, 17],  # "əɹ`"
    ]
    value = [
        70,  # "p⁼wo"
        68,  # "p⁼"
        74,  # "t⁼"
        76,  # "k⁼"
        79,  # "tʃ⁼"
        91,  # "ts`⁼ɹ`"
        85,  # "ts`⁼"
        92,  # "ts`ʰɹ`"
        86,  # "ts`ʰ"
        89,  # "s`ɹ`"
        87,  # "s`"
        90,  # "ɹ`ɹ`"
        88,  # "ɹ`"
        93,  # "ts⁼ɹ"
        82,  # "ts⁼"
        113,  # "əɹ`"
    ]
    token_str = ",".join(map(str, phoneme_tokens))
    new_lst_str = []
    for idx, sub in enumerate(sublist):
        sub_str = "," + ",".join(map(str, sub)) + ","
        if sub_str in token_str:
            replace_str = "," + str(value[idx]) + ","
            token_str = token_str.replace(sub_str, replace_str)

    new_lst = list(map(int, token_str.split(",")))
    return new_lst


def _connect_tone(phoneme_tokens, vocab):
    tone_list = ["→", "↑", "↓↑", "↓"]
    tone_token = []
    last_single_token = 0
    base = 0
    pattern = r"\[[^\[\]]*\]"  # Exclude "[" and "]"
    for tone, idx in vocab.items():
        if re.match(pattern, tone):
            base = idx + 1
        if tone in tone_list:
            tone_token.append(idx)
            last_single_token = idx

    pre_token = None
    cur_token = None
    res_token = []
    for t in phoneme_tokens:
        cur_token = t
        if t in tone_token:
            cur_token = (
                last_single_token
                + (pre_token - base) * len(tone_list)
                + tone_token.index(t)
                + 1
            )
            res_token.pop()
        res_token.append(cur_token)
        pre_token = t

    return res_token
