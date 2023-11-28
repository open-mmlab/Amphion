# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This module is modified from [Whisper](https://github.com/openai/whisper.git).

# ## Citations

# ```bibtex
# @inproceedings{openai-whisper,
#   author       = {Alec Radford and
#                   Jong Wook Kim and
#                   Tao Xu and
#                   Greg Brockman and
#                   Christine McLeavey and
#                   Ilya Sutskever},
#   title        = {Robust Speech Recognition via Large-Scale Weak Supervision},
#   booktitle    = {{ICML}},
#   series       = {Proceedings of Machine Learning Research},
#   volume       = {202},
#   pages        = {28492--28518},
#   publisher    = {{PMLR}},
#   year         = {2023}
# }
# ```
#

import re
import unicodedata

import regex

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s
