# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from utils.g2p.japanese import japanese_to_ipa
from utils.g2p.mandarin import chinese_to_ipa
from utils.g2p.english import english_to_ipa
from utils.g2p.french import french_to_ipa
from utils.g2p.korean import korean_to_ipa
from utils.g2p.german import german_to_ipa

patterns = [
    r"\[EN\](.*?)\[EN\]",
    r"\[ZH\](.*?)\[ZH\]",
    r"\[JA\](.*?)\[JA\]",
    r"\[FR\](.*?)\[FR\]",
    r"\[KR\](.*?)\[KR\]",
    r"\[DE\](.*?)\[DE\]",
]


def cje_cleaners(text):
    matches = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text))

    matches.sort(key=lambda x: x.start())  # Sort matches by their start positions

    outputs = ""
    for match in matches:
        text_segment = text[match.start() : match.end()]
        phone = clean_one(text_segment)
        outputs += phone

    return outputs


def clean_one(text):
    if text.find("[ZH]") != -1:
        text = re.sub(
            r"\[ZH\](.*?)\[ZH\]", lambda x: chinese_to_ipa(x.group(1)) + " ", text
        )
    if text.find("[JA]") != -1:
        text = re.sub(
            r"\[JA\](.*?)\[JA\]", lambda x: japanese_to_ipa(x.group(1)) + " ", text
        )
    if text.find("[EN]") != -1:
        text = re.sub(
            r"\[EN\](.*?)\[EN\]", lambda x: english_to_ipa(x.group(1)) + " ", text
        )
    if text.find("[FR]") != -1:
        text = re.sub(
            r"\[FR\](.*?)\[FR\]", lambda x: french_to_ipa(x.group(1)) + " ", text
        )
    if text.find("[KR]") != -1:
        text = re.sub(
            r"\[KR\](.*?)\[KR\]", lambda x: korean_to_ipa(x.group(1)) + " ", text
        )
    if text.find("[DE]") != -1:
        text = re.sub(
            r"\[DE\](.*?)\[DE\]", lambda x: german_to_ipa(x.group(1)) + " ", text
        )
    text = re.sub(r"\s+$", "", text)
    text = re.sub(r"([^\.,!\?\-…~])$", r"\1.", text)
    return text
