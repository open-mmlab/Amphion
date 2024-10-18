# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import jieba
import cn2an
from pypinyin import lazy_pinyin, BOPOMOFO
from typing import List
from models.tts.maskgct.g2p.g2p.chinese_model_g2p import BertPolyPredict
from models.tts.maskgct.g2p.utils.front_utils import *
import os

# from g2pw import G2PWConverter


# set blank level, {0："none",1:"char", 2:"word"}
BLANK_LEVEL = 0

# conv = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)
resource_path = r"./models/tts/maskgct/g2p"
poly_all_class_path = os.path.join(
    resource_path, "sources", "g2p_chinese_model", "polychar.txt"
)
if not os.path.exists(poly_all_class_path):
    print(
        "Incorrect path for polyphonic character class dictionary: {}, please check...".format(
            poly_all_class_path
        )
    )
    exit()
poly_dict = generate_poly_lexicon(poly_all_class_path)

# Set up G2PW model parameters
g2pw_poly_model_path = os.path.join(resource_path, "sources", "g2p_chinese_model")
if not os.path.exists(g2pw_poly_model_path):
    print(
        "Incorrect path for g2pw polyphonic character model: {}, please check...".format(
            g2pw_poly_model_path
        )
    )
    exit()

json_file_path = os.path.join(
    resource_path, "sources", "g2p_chinese_model", "polydict.json"
)
if not os.path.exists(json_file_path):
    print(
        "Incorrect path for g2pw id to pinyin dictionary: {}, please check...".format(
            json_file_path
        )
    )
    exit()

jsonr_file_path = os.path.join(
    resource_path, "sources", "g2p_chinese_model", "polydict_r.json"
)
if not os.path.exists(jsonr_file_path):
    print(
        "Incorrect path for g2pw pinyin to id dictionary: {}, please check...".format(
            jsonr_file_path
        )
    )
    exit()

g2pw_poly_predict = BertPolyPredict(
    g2pw_poly_model_path, jsonr_file_path, json_file_path
)


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
must_not_er_words = {"女儿", "老儿", "男儿", "少儿", "小儿"}

word_pinyin_dict = {}
with open(
    r"./models/tts/maskgct/g2p/sources/chinese_lexicon.txt", "r", encoding="utf-8"
) as fread:
    txt_list = fread.readlines()
    for txt in txt_list:
        word, pinyin = txt.strip().split("\t")
        word_pinyin_dict[word] = pinyin
    fread.close()

pinyin_2_bopomofo_dict = {}
with open(
    r"./models/tts/maskgct/g2p/sources/pinyin_2_bpmf.txt", "r", encoding="utf-8"
) as fread:
    txt_list = fread.readlines()
    for txt in txt_list:
        pinyin, bopomofo = txt.strip().split("\t")
        pinyin_2_bopomofo_dict[pinyin] = bopomofo
    fread.close()

tone_dict = {
    "0": "˙",
    "5": "˙",
    "1": "",
    "2": "ˊ",
    "3": "ˇ",
    "4": "ˋ",
}

bopomofos2pinyin_dict = {}
with open(
    r"./models/tts/maskgct/g2p/sources/bpmf_2_pinyin.txt", "r", encoding="utf-8"
) as fread:
    txt_list = fread.readlines()
    for txt in txt_list:
        v, k = txt.strip().split("\t")
        bopomofos2pinyin_dict[k] = v
    fread.close()


def bpmf_to_pinyin(text):
    bopomofo_list = text.split("|")
    pinyin_list = []
    for info in bopomofo_list:
        pinyin = ""
        for c in info:
            if c in bopomofos2pinyin_dict:
                pinyin += bopomofos2pinyin_dict[c]
        if len(pinyin) == 0:
            continue
        if pinyin[-1] not in "01234":
            pinyin += "1"
        if pinyin[:-1] == "ve":
            pinyin = "y" + pinyin
        if pinyin[:-1] == "sh":
            pinyin = pinyin[:-1] + "i" + pinyin[-1]
        if pinyin == "sh":
            pinyin = pinyin[:-1] + "i"
        if pinyin[:-1] == "s":
            pinyin = "si" + pinyin[-1]
        if pinyin[:-1] == "c":
            pinyin = "ci" + pinyin[-1]
        if pinyin[:-1] == "i":
            pinyin = "yi" + pinyin[-1]
        if pinyin[:-1] == "iou":
            pinyin = "you" + pinyin[-1]
        if pinyin[:-1] == "ien":
            pinyin = "yin" + pinyin[-1]
        if "iou" in pinyin and pinyin[-4:-1] == "iou":
            pinyin = pinyin[:-4] + "iu" + pinyin[-1]
        if "uei" in pinyin:
            if pinyin[:-1] == "uei":
                pinyin = "wei" + pinyin[-1]
            elif pinyin[-4:-1] == "uei":
                pinyin = pinyin[:-4] + "ui" + pinyin[-1]
        if "uen" in pinyin and pinyin[-4:-1] == "uen":
            if pinyin[:-1] == "uen":
                pinyin = "wen" + pinyin[-1]
            elif pinyin[-4:-1] == "uei":
                pinyin = pinyin[:-4] + "un" + pinyin[-1]
        if "van" in pinyin and pinyin[-4:-1] == "van":
            if pinyin[:-1] == "van":
                pinyin = "yuan" + pinyin[-1]
            elif pinyin[-4:-1] == "van":
                pinyin = pinyin[:-4] + "uan" + pinyin[-1]
        if "ueng" in pinyin and pinyin[-5:-1] == "ueng":
            pinyin = pinyin[:-5] + "ong" + pinyin[-1]
        if pinyin[:-1] == "veng":
            pinyin = "yong" + pinyin[-1]
        if "veng" in pinyin and pinyin[-5:-1] == "veng":
            pinyin = pinyin[:-5] + "iong" + pinyin[-1]
        if pinyin[:-1] == "ieng":
            pinyin = "ying" + pinyin[-1]
        if pinyin[:-1] == "u":
            pinyin = "wu" + pinyin[-1]
        if pinyin[:-1] == "v":
            pinyin = "yv" + pinyin[-1]
        if pinyin[:-1] == "ing":
            pinyin = "ying" + pinyin[-1]
        if pinyin[:-1] == "z":
            pinyin = "zi" + pinyin[-1]
        if pinyin[:-1] == "zh":
            pinyin = "zhi" + pinyin[-1]
        if pinyin[0] == "u":
            pinyin = "w" + pinyin[1:]
        if pinyin[0] == "i":
            pinyin = "y" + pinyin[1:]
        pinyin = pinyin.replace("ien", "in")

        pinyin_list.append(pinyin)
    return " ".join(pinyin_list)


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


def change_tone(bopomofo: str, tone: str) -> str:
    if bopomofo[-1] not in "˙ˊˇˋ":
        bopomofo = bopomofo + tone
    else:
        bopomofo = bopomofo[:-1] + tone
    return bopomofo


def er_sandhi(word: str, bopomofos: List[str]) -> List[str]:
    if len(word) > 1 and word[-1] == "儿" and word not in must_not_er_words:
        bopomofos[-1] = change_tone(bopomofos[-1], "˙")
    return bopomofos


def bu_sandhi(word: str, bopomofos: List[str]) -> List[str]:
    valid_char = set(word)
    if len(valid_char) == 1 and "不" in valid_char:
        pass
    elif word in ["不字"]:
        pass
    elif len(word) == 3 and word[1] == "不" and bopomofos[1][:-1] == "ㄅㄨ":
        bopomofos[1] = bopomofos[1][:-1] + "˙"
    else:
        for i, char in enumerate(word):
            if (
                i + 1 < len(bopomofos)
                and char == "不"
                and i + 1 < len(word)
                and 0 < len(bopomofos[i + 1])
                and bopomofos[i + 1][-1] == "ˋ"
            ):
                bopomofos[i] = bopomofos[i][:-1] + "ˊ"
    return bopomofos


def yi_sandhi(word: str, bopomofos: List[str]) -> List[str]:
    punc = "：，；。？！“”‘’':,;.?!()（）{}【】[]-~`、 "
    if word.find("一") != -1 and any(
        [item.isnumeric() for item in word if item != "一"]
    ):
        for i in range(len(word)):
            if (
                i == 0
                and word[0] == "一"
                and len(word) > 1
                and word[1]
                not in [
                    "零",
                    "一",
                    "二",
                    "三",
                    "四",
                    "五",
                    "六",
                    "七",
                    "八",
                    "九",
                    "十",
                ]
            ):
                if len(bopomofos[0]) > 0 and bopomofos[1][-1] in ["ˋ", "˙"]:
                    bopomofos[0] = change_tone(bopomofos[0], "ˊ")
                else:
                    bopomofos[0] = change_tone(bopomofos[0], "ˋ")
            elif word[i] == "一":
                bopomofos[i] = change_tone(bopomofos[i], "")
        return bopomofos
    elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
        bopomofos[1] = change_tone(bopomofos[1], "˙")
    elif word.startswith("第一"):
        bopomofos[1] = change_tone(bopomofos[1], "")
    elif word.startswith("一月") or word.startswith("一日") or word.startswith("一号"):
        bopomofos[0] = change_tone(bopomofos[0], "")
    else:
        for i, char in enumerate(word):
            if char == "一" and i + 1 < len(word):
                if (
                    len(bopomofos) > i + 1
                    and len(bopomofos[i + 1]) > 0
                    and bopomofos[i + 1][-1] in {"ˋ"}
                ):
                    bopomofos[i] = change_tone(bopomofos[i], "ˊ")
                else:
                    if word[i + 1] not in punc:
                        bopomofos[i] = change_tone(bopomofos[i], "ˋ")
                    else:
                        pass
    return bopomofos


def merge_bu(seg: List) -> List:
    new_seg = []
    last_word = ""
    for word in seg:
        if word != "不":
            if last_word == "不":
                word = last_word + word
            new_seg.append(word)
        last_word = word
    return new_seg


def merge_er(seg: List) -> List:
    new_seg = []
    for i, word in enumerate(seg):
        if i - 1 >= 0 and word == "儿":
            new_seg[-1] = new_seg[-1] + seg[i]
        else:
            new_seg.append(word)
    return new_seg


def merge_yi(seg: List) -> List:
    new_seg = []
    # function 1
    for i, word in enumerate(seg):
        if (
            i - 1 >= 0
            and word == "一"
            and i + 1 < len(seg)
            and seg[i - 1] == seg[i + 1]
        ):
            if i - 1 < len(new_seg):
                new_seg[i - 1] = new_seg[i - 1] + "一" + new_seg[i - 1]
            else:
                new_seg.append(word)
                new_seg.append(seg[i + 1])
        else:
            if i - 2 >= 0 and seg[i - 1] == "一" and seg[i - 2] == word:
                continue
            else:
                new_seg.append(word)
    seg = new_seg
    new_seg = []
    isnumeric_flag = False
    for i, word in enumerate(seg):
        if all([item.isnumeric() for item in word]) and not isnumeric_flag:
            isnumeric_flag = True
            new_seg.append(word)
        else:
            new_seg.append(word)
    seg = new_seg
    new_seg = []
    # function 2
    for i, word in enumerate(seg):
        if new_seg and new_seg[-1] == "一":
            new_seg[-1] = new_seg[-1] + word
        else:
            new_seg.append(word)
    return new_seg


# Word Segmentation, and convert Chinese pronunciation to pinyin (bopomofo)
def chinese_to_bopomofo(text_short, sentence):
    # bopomofos = conv(text_short)
    words = jieba.lcut(text_short, cut_all=False)
    words = merge_yi(words)
    words = merge_bu(words)
    words = merge_er(words)
    text = ""

    char_index = 0
    for word in words:
        bopomofos = []
        if word in word_pinyin_dict and word not in poly_dict:
            pinyin = word_pinyin_dict[word]
            for py in pinyin.split(" "):
                if py[:-1] in pinyin_2_bopomofo_dict and py[-1] in tone_dict:
                    bopomofos.append(
                        pinyin_2_bopomofo_dict[py[:-1]] + tone_dict[py[-1]]
                    )
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
                else:
                    bopomofos_lazy = lazy_pinyin(word, BOPOMOFO)
                    bopomofos += bopomofos_lazy
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
        else:
            for i in range(len(word)):
                c = word[i]
                if c in poly_dict:
                    poly_pinyin = g2pw_poly_predict.predict_process(
                        [text_short, char_index + i]
                    )[0]
                    py = poly_pinyin[2:-1]
                    bopomofos.append(
                        pinyin_2_bopomofo_dict[py[:-1]] + tone_dict[py[-1]]
                    )
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
                elif c in word_pinyin_dict:
                    py = word_pinyin_dict[c]
                    bopomofos.append(
                        pinyin_2_bopomofo_dict[py[:-1]] + tone_dict[py[-1]]
                    )
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
                else:
                    bopomofos.append(c)
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
        if BLANK_LEVEL == 2:
            bopomofos.append("_")
        char_index += len(word)

        if (
            len(word) == 3
            and bopomofos[0][-1] == "ˇ"
            and bopomofos[1][-1] == "ˇ"
            and bopomofos[-1][-1] == "ˇ"
        ):
            bopomofos[0] = bopomofos[0] + "ˊ"
            bopomofos[1] = bopomofos[1] + "ˊ"
        if len(word) == 2 and bopomofos[0][-1] == "ˇ" and bopomofos[-1][-1] == "ˇ":
            bopomofos[0] = bopomofos[0][:-1] + "ˊ"
        bopomofos = bu_sandhi(word, bopomofos)
        bopomofos = yi_sandhi(word, bopomofos)
        bopomofos = er_sandhi(word, bopomofos)
        if not re.search("[\u4e00-\u9fff]", word):
            text += "|" + word
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


def _chinese_to_ipa(text, sentence):
    text = number_to_chinese(text.strip())
    text = normalization(text)
    text = chinese_to_bopomofo(text, sentence)
    # pinyin = bpmf_to_pinyin(text)
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
def chinese_to_ipa(text, sentence, text_tokenizer):
    # phonemes = text_tokenizer(text.strip())
    if type(text) == str:
        return _chinese_to_ipa(text, sentence)
    else:
        result_ph = []
        for t in text:
            result_ph.append(_chinese_to_ipa(t, sentence))
        return result_ph
