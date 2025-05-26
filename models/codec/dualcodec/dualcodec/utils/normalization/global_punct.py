# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re

punct_map = {
    "　": " ",
    "！": "!",
    "＂": '"',
    "＃": "#",
    "＄": "$",
    "％": "%",
    "＆": "&",
    "＇": "'",
    "（": "(",
    "）": ")",
    "＊": "*",
    "＋": "+",
    "，": ",",
    "－": "-",
    "．": ".",
    "／": "/",
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
    "：": ":",
    "；": ";",
    "＜": "<",
    "＝": "=",
    "＞": ">",
    "？": "?",
    "＠": "@",
    "Ａ": "A",
    "Ｂ": "B",
    "Ｃ": "C",
    "Ｄ": "D",
    "Ｅ": "E",
    "Ｆ": "F",
    "Ｇ": "G",
    "Ｈ": "H",
    "Ｉ": "I",
    "Ｊ": "J",
    "Ｋ": "K",
    "Ｌ": "L",
    "Ｍ": "M",
    "Ｎ": "N",
    "Ｏ": "O",
    "Ｐ": "P",
    "Ｑ": "Q",
    "Ｒ": "R",
    "Ｓ": "S",
    "Ｔ": "T",
    "Ｕ": "U",
    "Ｖ": "V",
    "Ｗ": "W",
    "Ｘ": "X",
    "Ｙ": "Y",
    "Ｚ": "Z",
    "［": "[",
    "＼": "\\",
    "］": "]",
    "＾": "^",
    "＿": "_",
    "｀": "`",
    "ａ": "a",
    "ｂ": "b",
    "ｃ": "c",
    "ｄ": "d",
    "ｅ": "e",
    "ｆ": "f",
    "ｇ": "g",
    "ｈ": "h",
    "ｉ": "i",
    "ｊ": "j",
    "ｋ": "k",
    "ｌ": "l",
    "ｍ": "m",
    "ｎ": "n",
    "ｏ": "o",
    "ｐ": "p",
    "ｑ": "q",
    "ｒ": "r",
    "ｓ": "s",
    "ｔ": "t",
    "ｕ": "u",
    "ｖ": "v",
    "ｗ": "w",
    "ｘ": "x",
    "ｙ": "y",
    "ｚ": "z",
    "｛": "{",
    "｜": "|",
    "｝": "}",
    # Custom symbol mappings
    "。": ".",  # Period
    "《": '"',
    "》": '"',
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "、": ",",
    "・": "-",
    "【": '"',
    "】": '"',  # Square brackets
    "「": '"',
    "」": '"',  # Japanese quotation marks
    "『": '"',
    "』": '"',  # Japanese book title marks
    "—": "...",
    "…": "...",  # Ellipsis and dash
}


def normalize_punctuation(text):
    """
    Normalize punctuation in text by the mapping in punct_map

    Args:
        text (str): Text to be normalized
    """
    for original, replacement in punct_map.items():
        text = text.replace(original, replacement)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    text = text.replace("(", ",")
    text = text.replace(")", ",")
    text = text.replace(";", ",")
    text = text.replace(":", ",")

    return text
