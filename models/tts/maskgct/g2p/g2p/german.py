# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

"""
    Text clean time
"""
rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": ".",
    "…": ".",
    "$": ".",
    "“": "",
    "”": "",
    "‘": "",
    "’": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "《": "",
    "》": "",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "—": "",
    "～": "-",
    "~": "-",
    "「": "",
    "」": "",
    "¿": "",
    "¡": "",
}


def collapse_whitespace(text):
    # Regular expression matching whitespace:
    _whitespace_re = re.compile(r"\s+")
    return re.sub(_whitespace_re, " ", text).strip()


def remove_punctuation_at_begin(text):
    return re.sub(r"^[,.!?]+", "", text)


def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    return text


def replace_symbols(text):
    text = text.replace(";", ",")
    text = text.replace("-", " ")
    text = text.replace(":", ",")
    return text


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text


def text_normalize(text):
    text = replace_punctuation(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = remove_punctuation_at_begin(text)
    text = collapse_whitespace(text)
    text = re.sub(r"([^\.,!\?\-…])$", r"\1", text)
    return text


def german_to_ipa(text, text_tokenizer):
    if type(text) == str:
        text = text_normalize(text)
        phonemes = text_tokenizer(text)
        return phonemes
    else:
        for i, t in enumerate(text):
            text[i] = text_normalize(t)
        return text_tokenizer(text)
