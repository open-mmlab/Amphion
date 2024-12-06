# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import jieba
import cn2an

"""
    Text clean time
"""
# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "ㄟˉ"),
        ("b", "ㄅㄧˋ"),
        ("c", "ㄙㄧˉ"),
        ("d", "ㄉㄧˋ"),
        ("e", "ㄧˋ"),
        ("f", "ㄝˊㄈㄨˋ"),
        ("g", "ㄐㄧˋ"),
        ("h", "ㄝˇㄑㄩˋ"),
        ("i", "ㄞˋ"),
        ("j", "ㄐㄟˋ"),
        ("k", "ㄎㄟˋ"),
        ("l", "ㄝˊㄛˋ"),
        ("m", "ㄝˊㄇㄨˋ"),
        ("n", "ㄣˉ"),
        ("o", "ㄡˉ"),
        ("p", "ㄆㄧˉ"),
        ("q", "ㄎㄧㄡˉ"),
        ("r", "ㄚˋ"),
        ("s", "ㄝˊㄙˋ"),
        ("t", "ㄊㄧˋ"),
        ("u", "ㄧㄡˉ"),
        ("v", "ㄨㄧˉ"),
        ("w", "ㄉㄚˋㄅㄨˋㄌㄧㄡˋ"),
        ("x", "ㄝˉㄎㄨˋㄙˋ"),
        ("y", "ㄨㄞˋ"),
        ("z", "ㄗㄟˋ"),
    ]
]

# List of (bopomofo, ipa) pairs:
_bopomofo_to_ipa = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("ㄅㄛ", "p⁼wo"),
        ("ㄆㄛ", "pʰwo"),
        ("ㄇㄛ", "mwo"),
        ("ㄈㄛ", "fwo"),
        ("ㄧㄢ", "|jɛn"),
        ("ㄩㄢ", "|ɥæn"),
        ("ㄧㄣ", "|in"),
        ("ㄩㄣ", "|ɥn"),
        ("ㄧㄥ", "|iŋ"),
        ("ㄨㄥ", "|ʊŋ"),
        ("ㄩㄥ", "|jʊŋ"),
        # Add
        ("ㄧㄚ", "|ia"),
        ("ㄧㄝ", "|iɛ"),
        ("ㄧㄠ", "|iɑʊ"),
        ("ㄧㄡ", "|ioʊ"),
        ("ㄧㄤ", "|iɑŋ"),
        ("ㄨㄚ", "|ua"),
        ("ㄨㄛ", "|uo"),
        ("ㄨㄞ", "|uaɪ"),
        ("ㄨㄟ", "|ueɪ"),
        ("ㄨㄢ", "|uan"),
        ("ㄨㄣ", "|uən"),
        ("ㄨㄤ", "|uɑŋ"),
        ("ㄩㄝ", "|ɥɛ"),
        # End
        ("ㄅ", "p⁼"),
        ("ㄆ", "pʰ"),
        ("ㄇ", "m"),
        ("ㄈ", "f"),
        ("ㄉ", "t⁼"),
        ("ㄊ", "tʰ"),
        ("ㄋ", "n"),
        ("ㄌ", "l"),
        ("ㄍ", "k⁼"),
        ("ㄎ", "kʰ"),
        ("ㄏ", "x"),
        ("ㄐ", "tʃ⁼"),
        ("ㄑ", "tʃʰ"),
        ("ㄒ", "ʃ"),
        ("ㄓ", "ts`⁼"),
        ("ㄔ", "ts`ʰ"),
        ("ㄕ", "s`"),
        ("ㄖ", "ɹ`"),
        ("ㄗ", "ts⁼"),
        ("ㄘ", "tsʰ"),
        ("ㄙ", "|s"),
        ("ㄚ", "|a"),
        ("ㄛ", "|o"),
        ("ㄜ", "|ə"),
        ("ㄝ", "|ɛ"),
        ("ㄞ", "|aɪ"),
        ("ㄟ", "|eɪ"),
        ("ㄠ", "|ɑʊ"),
        ("ㄡ", "|oʊ"),
        ("ㄢ", "|an"),
        ("ㄣ", "|ən"),
        ("ㄤ", "|ɑŋ"),
        ("ㄥ", "|əŋ"),
        ("ㄦ", "əɹ"),
        ("ㄧ", "|i"),
        ("ㄨ", "|u"),
        ("ㄩ", "|ɥ"),
        ("ˉ", "→|"),
        ("ˊ", "↑|"),
        ("ˇ", "↓↑|"),
        ("ˋ", "↓|"),
        ("˙", "|"),
    ]
]


# Convert numbers to Chinese pronunciation
def number_to_chinese(text):
    # numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    # for number in numbers:
    #     text = text.replace(number, cn2an.an2cn(number), 1)
    text = cn2an.transform(text, "an2cn")
    return text


def normalization(text):
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("！", "!")
    text = text.replace("？", "?")
    text = text.replace("；", ";")
    text = text.replace("：", ":")
    text = text.replace("、", ",")
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("⋯", "…")
    text = text.replace("···", "…")
    text = text.replace("・・・", "…")
    text = text.replace("...", "…")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff\s_,\.\?!;:\'…]", "", text)
    text = re.sub(r"\s*([,\.\?!;:\'…])\s*", r"\1", text)
    return text


# Word Segmentation, and convert Chinese pronunciation to pinyin (bopomofo)
def chinese_to_bopomofo(text):
    from pypinyin import lazy_pinyin, BOPOMOFO

    words = jieba.lcut(text, cut_all=False)
    text = ""
    for word in words:
        bopomofos = lazy_pinyin(word, BOPOMOFO)
        if not re.search("[\u4e00-\u9fff]", word):
            text += word
            continue
        for i in range(len(bopomofos)):
            bopomofos[i] = re.sub(r"([\u3105-\u3129])$", r"\1ˉ", bopomofos[i])
        if text != "":
            text += "|"
        text += "|".join(bopomofos)
    return text


# Convert latin pronunciation to pinyin (bopomofo)
def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text


# Convert pinyin (bopomofo) to IPA
def bopomofo_to_ipa(text):
    for regex, replacement in _bopomofo_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


def _chinese_to_ipa(text):
    text = number_to_chinese(text.strip())
    text = normalization(text)
    # print("Normalized text: ", text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_ipa(text)
    text = re.sub("([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ\2", text)
    text = re.sub("([s][⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ\2", text)
    text = re.sub(r"^\||[^\w\s_,\.\?!;:\'…\|→↓↑⁼ʰ`]", "", text)
    text = re.sub(r"([,\.\?!;:\'…])", r"|\1|", text)
    text = re.sub(r"\|+", "|", text)
    text = text.rstrip("|")
    return text


# Convert Chinese to IPA
def chinese_to_ipa(text, text_tokenizer):
    # phonemes = text_tokenizer(text.strip())
    if type(text) == str:
        return _chinese_to_ipa(text)
    else:
        result_ph = []
        for t in text:
            result_ph.append(_chinese_to_ipa(t))
        return result_ph
