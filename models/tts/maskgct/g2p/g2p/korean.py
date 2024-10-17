# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

"""
    Text clean time
"""
english_dictionary = {
    "KOREA": "코리아",
    "IDOL": "아이돌",
    "IT": "아이티",
    "IQ": "아이큐",
    "UP": "업",
    "DOWN": "다운",
    "PC": "피씨",
    "CCTV": "씨씨티비",
    "SNS": "에스엔에스",
    "AI": "에이아이",
    "CEO": "씨이오",
    "A": "에이",
    "B": "비",
    "C": "씨",
    "D": "디",
    "E": "이",
    "F": "에프",
    "G": "지",
    "H": "에이치",
    "I": "아이",
    "J": "제이",
    "K": "케이",
    "L": "엘",
    "M": "엠",
    "N": "엔",
    "O": "오",
    "P": "피",
    "Q": "큐",
    "R": "알",
    "S": "에스",
    "T": "티",
    "U": "유",
    "V": "브이",
    "W": "더블유",
    "X": "엑스",
    "Y": "와이",
    "Z": "제트",
}


def normalize(text):
    text = text.strip()
    text = re.sub(
        "[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]", "", text
    )
    text = normalize_english(text)
    text = text.lower()
    return text


def normalize_english(text):
    def fn(m):
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text


def korean_to_ipa(text, text_tokenizer):
    if type(text) == str:
        text = normalize(text)
        phonemes = text_tokenizer(text)
        return phonemes
    else:
        for i, t in enumerate(text):
            text[i] = normalize(t)
        return text_tokenizer(text)
