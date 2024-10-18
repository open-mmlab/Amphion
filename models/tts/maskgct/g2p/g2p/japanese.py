# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io, re, os, sys, time, argparse, pdb, json
from io import StringIO
from typing import Optional
import numpy as np
import traceback
import pyopenjtalk
from pykakasi import kakasi

punctuation = [",", ".", "!", "?", ":", ";", "'", "…"]

jp_xphone2ipa = [
    " a a",
    " i i",
    " u ɯ",
    " e e",
    " o o",
    " a: aː",
    " i: iː",
    " u: ɯː",
    " e: eː",
    " o: oː",
    " k k",
    " s s",
    " t t",
    " n n",
    " h ç",
    " f ɸ",
    " m m",
    " y j",
    " r ɾ",
    " w ɰᵝ",
    " N ɴ",
    " g g",
    " j d ʑ",
    " z z",
    " d d",
    " b b",
    " p p",
    " q q",
    " v v",
    " : :",
    " by b j",
    " ch t ɕ",
    " dy d e j",
    " ty t e j",
    " gy g j",
    " gw g ɯ",
    " hy ç j",
    " ky k j",
    " kw k ɯ",
    " my m j",
    " ny n j",
    " py p j",
    " ry ɾ j",
    " sh ɕ",
    " ts t s ɯ",
]

_mora_list_minimum: list[tuple[str, Optional[str], str]] = [
    ("ヴォ", "v", "o"),
    ("ヴェ", "v", "e"),
    ("ヴィ", "v", "i"),
    ("ヴァ", "v", "a"),
    ("ヴ", "v", "u"),
    ("ン", None, "N"),
    ("ワ", "w", "a"),
    ("ロ", "r", "o"),
    ("レ", "r", "e"),
    ("ル", "r", "u"),
    ("リョ", "ry", "o"),
    ("リュ", "ry", "u"),
    ("リャ", "ry", "a"),
    ("リェ", "ry", "e"),
    ("リ", "r", "i"),
    ("ラ", "r", "a"),
    ("ヨ", "y", "o"),
    ("ユ", "y", "u"),
    ("ヤ", "y", "a"),
    ("モ", "m", "o"),
    ("メ", "m", "e"),
    ("ム", "m", "u"),
    ("ミョ", "my", "o"),
    ("ミュ", "my", "u"),
    ("ミャ", "my", "a"),
    ("ミェ", "my", "e"),
    ("ミ", "m", "i"),
    ("マ", "m", "a"),
    ("ポ", "p", "o"),
    ("ボ", "b", "o"),
    ("ホ", "h", "o"),
    ("ペ", "p", "e"),
    ("ベ", "b", "e"),
    ("ヘ", "h", "e"),
    ("プ", "p", "u"),
    ("ブ", "b", "u"),
    ("フォ", "f", "o"),
    ("フェ", "f", "e"),
    ("フィ", "f", "i"),
    ("ファ", "f", "a"),
    ("フ", "f", "u"),
    ("ピョ", "py", "o"),
    ("ピュ", "py", "u"),
    ("ピャ", "py", "a"),
    ("ピェ", "py", "e"),
    ("ピ", "p", "i"),
    ("ビョ", "by", "o"),
    ("ビュ", "by", "u"),
    ("ビャ", "by", "a"),
    ("ビェ", "by", "e"),
    ("ビ", "b", "i"),
    ("ヒョ", "hy", "o"),
    ("ヒュ", "hy", "u"),
    ("ヒャ", "hy", "a"),
    ("ヒェ", "hy", "e"),
    ("ヒ", "h", "i"),
    ("パ", "p", "a"),
    ("バ", "b", "a"),
    ("ハ", "h", "a"),
    ("ノ", "n", "o"),
    ("ネ", "n", "e"),
    ("ヌ", "n", "u"),
    ("ニョ", "ny", "o"),
    ("ニュ", "ny", "u"),
    ("ニャ", "ny", "a"),
    ("ニェ", "ny", "e"),
    ("ニ", "n", "i"),
    ("ナ", "n", "a"),
    ("ドゥ", "d", "u"),
    ("ド", "d", "o"),
    ("トゥ", "t", "u"),
    ("ト", "t", "o"),
    ("デョ", "dy", "o"),
    ("デュ", "dy", "u"),
    ("デャ", "dy", "a"),
    # ("デェ", "dy", "e"),
    ("ディ", "d", "i"),
    ("デ", "d", "e"),
    ("テョ", "ty", "o"),
    ("テュ", "ty", "u"),
    ("テャ", "ty", "a"),
    ("ティ", "t", "i"),
    ("テ", "t", "e"),
    ("ツォ", "ts", "o"),
    ("ツェ", "ts", "e"),
    ("ツィ", "ts", "i"),
    ("ツァ", "ts", "a"),
    ("ツ", "ts", "u"),
    ("ッ", None, "q"),  # 「cl」から「q」に変更
    ("チョ", "ch", "o"),
    ("チュ", "ch", "u"),
    ("チャ", "ch", "a"),
    ("チェ", "ch", "e"),
    ("チ", "ch", "i"),
    ("ダ", "d", "a"),
    ("タ", "t", "a"),
    ("ゾ", "z", "o"),
    ("ソ", "s", "o"),
    ("ゼ", "z", "e"),
    ("セ", "s", "e"),
    ("ズィ", "z", "i"),
    ("ズ", "z", "u"),
    ("スィ", "s", "i"),
    ("ス", "s", "u"),
    ("ジョ", "j", "o"),
    ("ジュ", "j", "u"),
    ("ジャ", "j", "a"),
    ("ジェ", "j", "e"),
    ("ジ", "j", "i"),
    ("ショ", "sh", "o"),
    ("シュ", "sh", "u"),
    ("シャ", "sh", "a"),
    ("シェ", "sh", "e"),
    ("シ", "sh", "i"),
    ("ザ", "z", "a"),
    ("サ", "s", "a"),
    ("ゴ", "g", "o"),
    ("コ", "k", "o"),
    ("ゲ", "g", "e"),
    ("ケ", "k", "e"),
    ("グヮ", "gw", "a"),
    ("グ", "g", "u"),
    ("クヮ", "kw", "a"),
    ("ク", "k", "u"),
    ("ギョ", "gy", "o"),
    ("ギュ", "gy", "u"),
    ("ギャ", "gy", "a"),
    ("ギェ", "gy", "e"),
    ("ギ", "g", "i"),
    ("キョ", "ky", "o"),
    ("キュ", "ky", "u"),
    ("キャ", "ky", "a"),
    ("キェ", "ky", "e"),
    ("キ", "k", "i"),
    ("ガ", "g", "a"),
    ("カ", "k", "a"),
    ("オ", None, "o"),
    ("エ", None, "e"),
    ("ウォ", "w", "o"),
    ("ウェ", "w", "e"),
    ("ウィ", "w", "i"),
    ("ウ", None, "u"),
    ("イェ", "y", "e"),
    ("イ", None, "i"),
    ("ア", None, "a"),
]

_mora_list_additional: list[tuple[str, Optional[str], str]] = [
    ("ヴョ", "by", "o"),
    ("ヴュ", "by", "u"),
    ("ヴャ", "by", "a"),
    ("ヲ", None, "o"),
    ("ヱ", None, "e"),
    ("ヰ", None, "i"),
    ("ヮ", "w", "a"),
    ("ョ", "y", "o"),
    ("ュ", "y", "u"),
    ("ヅ", "z", "u"),
    ("ヂ", "j", "i"),
    ("ヶ", "k", "e"),
    ("ャ", "y", "a"),
    ("ォ", None, "o"),
    ("ェ", None, "e"),
    ("ゥ", None, "u"),
    ("ィ", None, "i"),
    ("ァ", None, "a"),
]

# 例: "vo" -> "ヴォ", "a" -> "ア"
mora_phonemes_to_mora_kata: dict[str, str] = {
    (consonant or "") + vowel: kana for [kana, consonant, vowel] in _mora_list_minimum
}

# 例: "ヴォ" -> ("v", "o"), "ア" -> (None, "a")
mora_kata_to_mora_phonemes: dict[str, tuple[Optional[str], str]] = {
    kana: (consonant, vowel)
    for [kana, consonant, vowel] in _mora_list_minimum + _mora_list_additional
}


# 正規化で記号を変換するための辞書
rep_map = {
    "：": ":",
    "；": ";",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "⋯": "…",
    "···": "…",
    "・・・": "…",
    "·": ",",
    "・": ",",
    "•": ",",
    "、": ",",
    "$": ".",
    # "“": "'",
    # "”": "'",
    # '"': "'",
    "‘": "'",
    "’": "'",
    # "（": "'",
    # "）": "'",
    # "(": "'",
    # ")": "'",
    # "《": "'",
    # "》": "'",
    # "【": "'",
    # "】": "'",
    # "[": "'",
    # "]": "'",
    # "——": "-",
    # "−": "-",
    # "-": "-",
    # "『": "'",
    # "』": "'",
    # "〈": "'",
    # "〉": "'",
    # "«": "'",
    # "»": "'",
    # # "～": "-",  # これは長音記号「ー」として扱うよう変更
    # # "~": "-",  # これは長音記号「ー」として扱うよう変更
    # "「": "'",
    # "」": "'",
}


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def replace_punctuation(text: str) -> str:
    """句読点等を「.」「,」「!」「?」「'」「-」に正規化し、OpenJTalkで読みが取得できるもののみ残す：
    漢字・平仮名・カタカナ、アルファベット、ギリシャ文字
    """
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    # print("before: ", text)
    # 句読点を辞書で置換
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        # ↓ ひらがな、カタカナ、漢字
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
        # ↓ 半角アルファベット（大文字と小文字）
        + r"\u0041-\u005A\u0061-\u007A"
        # ↓ 全角アルファベット（大文字と小文字）
        + r"\uFF21-\uFF3A\uFF41-\uFF5A"
        # ↓ ギリシャ文字
        + r"\u0370-\u03FF\u1F00-\u1FFF"
        # ↓ "!", "?", "…", ",", ".", "'", "-", 但し`…`はすでに`...`に変換されている
        + "".join(punctuation) + r"]+",
        # 上述以外の文字を削除
        "",
        replaced_text,
    )
    # print("after: ", replaced_text)
    return replaced_text


def fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone_list`のtone（アクセントの値）を0か1の範囲に修正する。
    例: [(a, 0), (i, -1), (u, -1)] → [(a, 1), (i, 0), (u, 0)]
    """
    tone_values = set(tone for _, tone in phone_tone_list)
    if len(tone_values) == 1:
        assert tone_values == {0}, tone_values
        return phone_tone_list
    elif len(tone_values) == 2:
        if tone_values == {0, 1}:
            return phone_tone_list
        elif tone_values == {-1, 0}:
            return [
                (letter, 0 if tone == -1 else 1) for letter, tone in phone_tone_list
            ]
        else:
            raise ValueError(f"Unexpected tone values: {tone_values}")
    else:
        raise ValueError(f"Unexpected tone values: {tone_values}")


def fix_phone_tone_wplen(phone_tone_list, word_phone_length_list):
    phones = []
    tones = []
    w_p_len = []
    p_len = len(phone_tone_list)
    idx = 0
    w_idx = 0
    while idx < p_len:
        offset = 0
        if phone_tone_list[idx] == "▁":
            w_p_len.append(w_idx + 1)

        curr_w_p_len = word_phone_length_list[w_idx]
        for i in range(curr_w_p_len):
            p, t = phone_tone_list[idx]
            if p == ":" and len(phones) > 0:
                if phones[-1][-1] != ":":
                    phones[-1] += ":"
                    offset -= 1
            else:
                phones.append(p)
                tones.append(str(t))
            idx += 1
            if idx >= p_len:
                break
        w_p_len.append(curr_w_p_len + offset)
        w_idx += 1
        # print(w_p_len)
    return phones, tones, w_p_len


def g2phone_tone_wo_punct(prosodies) -> list[tuple[str, int]]:
    """
    テキストに対して、音素とアクセント（0か1）のペアのリストを返す。
    ただし「!」「.」「?」等の非音素記号(punctuation)は全て消える（ポーズ記号も残さない）。
    非音素記号を含める処理は`align_tones()`で行われる。
    また「っ」は「cl」でなく「q」に変換される（「ん」は「N」のまま）。
    例: "こんにちは、世界ー。。元気？！" →
    [('k', 0), ('o', 0), ('N', 1), ('n', 1), ('i', 1), ('ch', 1), ('i', 1), ('w', 1), ('a', 1), ('s', 1), ('e', 1), ('k', 0), ('a', 0), ('i', 0), ('i', 0), ('g', 1), ('e', 1), ('N', 0), ('k', 0), ('i', 0)]
    """
    result: list[tuple[str, int]] = []
    current_phrase: list[tuple[str, int]] = []
    current_tone = 0
    last_accent = ""
    for i, letter in enumerate(prosodies):
        # 特殊記号の処理

        # 文頭記号、無視する
        if letter == "^":
            assert i == 0, "Unexpected ^"
        # アクセント句の終わりに来る記号
        elif letter in ("$", "?", "_", "#"):
            # 保持しているフレーズを、アクセント数値を0-1に修正し結果に追加
            result.extend(fix_phone_tone(current_phrase))
            # 末尾に来る終了記号、無視（文中の疑問文は`_`になる）
            if letter in ("$", "?"):
                assert i == len(prosodies) - 1, f"Unexpected {letter}"
            # あとは"_"（ポーズ）と"#"（アクセント句の境界）のみ
            # これらは残さず、次のアクセント句に備える。

            current_phrase = []
            # 0を基準点にしてそこから上昇・下降する（負の場合は上の`fix_phone_tone`で直る）
            current_tone = 0
            last_accent = ""
        # アクセント上昇記号
        elif letter == "[":
            if last_accent != letter:
                current_tone = current_tone + 1
            last_accent = letter
        # アクセント下降記号
        elif letter == "]":
            if last_accent != letter:
                current_tone = current_tone - 1
            last_accent = letter
        # それ以外は通常の音素
        else:
            if letter == "cl":  # 「っ」の処理
                letter = "q"
            current_phrase.append((letter, current_tone))
    return result


def handle_long(sep_phonemes: list[list[str]]) -> list[list[str]]:
    for i in range(len(sep_phonemes)):
        if sep_phonemes[i][0] == "ー":
            # sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
            sep_phonemes[i][0] = ":"
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    # sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
                    sep_phonemes[i][j] = ":"
    return sep_phonemes


def handle_long_word(sep_phonemes: list[list[str]]) -> list[list[str]]:
    res = []
    for i in range(len(sep_phonemes)):
        if sep_phonemes[i][0] == "ー":
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
            # sep_phonemes[i][0] = ':'
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
                    # sep_phonemes[i][j] = ':'
        res.append(sep_phonemes[i])
        res.append("▁")
    return res


def align_tones(
    phones_with_punct: list[str], phone_tone_list: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """
    例:
    …私は、、そう思う。
    phones_with_punct:
    [".", ".", ".", "w", "a", "t", "a", "sh", "i", "w", "a", ",", ",", "s", "o", "o", "o", "m", "o", "u", "."]
    phone_tone_list:
    [("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0))]
    Return:
    [(".", 0), (".", 0), (".", 0), ("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), (",", 0), (",", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0), (".", 0)]
    """
    result: list[tuple[str, int]] = []
    tone_index = 0
    for phone in phones_with_punct:
        if tone_index >= len(phone_tone_list):
            # 余ったpunctuationがある場合 → (punctuation, 0)を追加
            result.append((phone, 0))
        elif phone == phone_tone_list[tone_index][0]:
            # phone_tone_listの現在の音素と一致する場合 → toneをそこから取得、(phone, tone)を追加
            result.append((phone, phone_tone_list[tone_index][1]))
            # 探すindexを1つ進める
            tone_index += 1
        elif phone in punctuation or phone == "▁":
            # phoneがpunctuationの場合 → (phone, 0)を追加
            result.append((phone, 0))
        else:
            print(f"phones: {phones_with_punct}")
            print(f"phone_tone_list: {phone_tone_list}")
            print(f"result: {result}")
            print(f"tone_index: {tone_index}")
            print(f"phone: {phone}")
            raise ValueError(f"Unexpected phone: {phone}")
    return result


def kata2phoneme_list(text: str) -> list[str]:
    """
    原則カタカナの`text`を受け取り、それをそのままいじらずに音素記号のリストに変換。
    注意点：
    - punctuationが来た場合（punctuationが1文字の場合がありうる）、処理せず1文字のリストを返す
    - 冒頭に続く「ー」はそのまま「ー」のままにする（`handle_long()`で処理される）
    - 文中の「ー」は前の音素記号の最後の音素記号に変換される。
    例：
    `ーーソーナノカーー` → ["ー", "ー", "s", "o", "o", "n", "a", "n", "o", "k", "a", "a", "a"]
    `?` → ["?"]
    """
    if text in punctuation:
        return [text]
    # `text`がカタカナ（`ー`含む）のみからなるかどうかをチェック
    if re.fullmatch(r"[\u30A0-\u30FF]+", text) is None:
        raise ValueError(f"Input must be katakana only: {text}")
    sorted_keys = sorted(mora_kata_to_mora_phonemes.keys(), key=len, reverse=True)
    pattern = "|".join(map(re.escape, sorted_keys))

    def mora2phonemes(mora: str) -> str:
        cosonant, vowel = mora_kata_to_mora_phonemes[mora]
        if cosonant is None:
            return f" {vowel}"
        return f" {cosonant} {vowel}"

    spaced_phonemes = re.sub(pattern, lambda m: mora2phonemes(m.group()), text)

    # 長音記号「ー」の処理
    long_pattern = r"(\w)(ー*)"
    long_replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))
    spaced_phonemes = re.sub(long_pattern, long_replacement, spaced_phonemes)
    # spaced_phonemes += ' ▁'
    return spaced_phonemes.strip().split(" ")


def frontend2phoneme(labels, drop_unvoiced_vowels=False):
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]
        # print(lab_curr)
        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            # assert n == 0 or n == N - 1
            # if n == 0:
            #     phones.append("^")
            # elif n == N - 1:
            #     # check question form or not
            #     e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
            #     if e3 == 0:
            #         phones.append("$")
            #     elif e3 == 1:
            #         phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        # print(p3, a1, a2, a3, f1, a2_next, lab_curr)
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    # phones = ' '.join(phones)
    return phones


class JapanesePhoneConverter(object):
    def __init__(self, lexicon_path=None, ipa_dict_path=None):
        # lexicon_lines = open(lexicon_path, 'r', encoding='utf-8').readlines()
        # self.lexicon = {}
        # self.single_dict = {}
        # self.double_dict = {}
        # for curr_line in lexicon_lines:
        #     k,v = curr_line.strip().split('+',1)
        #     self.lexicon[k] = v
        #     if len(k) == 2:
        #         self.double_dict[k] = v
        #     elif len(k) == 1:
        #         self.single_dict[k] = v
        self.ipa_dict = {}
        for curr_line in jp_xphone2ipa:
            k, v = curr_line.strip().split(" ", 1)
            self.ipa_dict[k] = re.sub("\s", "", v)
        # kakasi1 = kakasi()
        # kakasi1.setMode("H","K")
        # kakasi1.setMode("J","K")
        # kakasi1.setMode("r","Hepburn")
        self.japan_JH2K = kakasi()
        self.table = {ord(f): ord(t) for f, t in zip("67", "_¯")}

    def text2sep_kata(self, parsed) -> tuple[list[str], list[str]]:
        """
        `text_normalize`で正規化済みの`norm_text`を受け取り、それを単語分割し、
        分割された単語リストとその読み（カタカナor記号1文字）のリストのタプルを返す。
        単語分割結果は、`g2p()`の`word2ph`で1文字あたりに割り振る音素記号の数を決めるために使う。
        例:
        `私はそう思う!って感じ?` →
        ["私", "は", "そう", "思う", "!", "って", "感じ", "?"], ["ワタシ", "ワ", "ソー", "オモウ", "!", "ッテ", "カンジ", "?"]
        """
        # parsed: OpenJTalkの解析結果
        sep_text: list[str] = []
        sep_kata: list[str] = []
        fix_parsed = []
        i = 0
        while i <= len(parsed) - 1:
            # word: 実際の単語の文字列
            # yomi: その読み、但し無声化サインの`’`は除去
            # print(parsed)
            yomi = parsed[i]["pron"]
            tmp_parsed = parsed[i]
            if i != len(parsed) - 1 and parsed[i + 1]["string"] in [
                "々",
                "ゝ",
                "ヽ",
                "ゞ",
                "ヾ",
                "゛",
            ]:
                word = parsed[i]["string"] + parsed[i + 1]["string"]
                i += 1
            else:
                word = parsed[i]["string"]
            word, yomi = replace_punctuation(word), yomi.replace("’", "")
            """
            ここで`yomi`の取りうる値は以下の通りのはず。
            - `word`が通常単語 → 通常の読み（カタカナ）
                （カタカナからなり、長音記号も含みうる、`アー` 等）
            - `word`が`ー` から始まる → `ーラー` や `ーーー` など
            - `word`が句読点や空白等 → `、`
            - `word`が`?` → `？`（全角になる）
            他にも`word`が読めないキリル文字アラビア文字等が来ると`、`になるが、正規化でこの場合は起きないはず。
            また元のコードでは`yomi`が空白の場合の処理があったが、これは起きないはず。
            処理すべきは`yomi`が`、`の場合のみのはず。
            """
            assert yomi != "", f"Empty yomi: {word}"
            if yomi == "、":
                # wordは正規化されているので、`.`, `,`, `!`, `'`, `-`のいずれか
                if word not in (
                    ".",
                    ",",
                    "!",
                    "'",
                    "-",
                    "?",
                    ":",
                    ";",
                    "…",
                    "",
                ):
                    # ここはpyopenjtalkが読めない文字等のときに起こる
                    print(
                        "Cannot read:{}, yomi:{}, new_word:{};".format(
                            word, yomi, self.japan_JH2K.convert(word)[0]["kana"]
                        )
                    )
                    # raise ValueError(word)
                    word = self.japan_JH2K.convert(word)[0]["kana"]
                    # print(word, self.japan_JH2K.convert(word)[0]['kana'], kata2phoneme_list(self.japan_JH2K.convert(word)[0]['kana']))
                    tmp_parsed["pron"] = word
                    # yomi = "-"
                    # word = ','
                # yomiは元の記号のままに変更
                # else:
                #     parsed[i]['pron'] = parsed[i]["string"]
                yomi = word
            elif yomi == "？":
                assert word == "?", f"yomi `？` comes from: {word}"
                yomi = "?"
            if word == "":
                i += 1
                continue
            sep_text.append(word)
            sep_kata.append(yomi)
            # print(word, yomi, parts)
            fix_parsed.append(tmp_parsed)
            i += 1
        # print(sep_text, sep_kata)
        return sep_text, sep_kata, fix_parsed

    def getSentencePhone(self, sentence, blank_mode=True, phoneme_mode=False):
        # print("origin:", sentence)
        words = []
        words_phone_len = []
        short_char_flag = False
        output_duration_flag = []
        output_before_sil_flag = []
        normed_text = []
        sentence = sentence.strip().strip("'")
        sentence = re.sub(r"\s+", "", sentence)
        output_res = []
        failed_words = []
        last_long_pause = 4
        last_word = None
        frontend_text = pyopenjtalk.run_frontend(sentence)
        # print("frontend_text: ", frontend_text)
        try:
            frontend_text = pyopenjtalk.estimate_accent(frontend_text)
        except:
            pass
        # print("estimate_accent: ", frontend_text)
        # sep_text: 単語単位の単語のリスト
        # sep_kata: 単語単位の単語のカタカナ読みのリスト
        sep_text, sep_kata, frontend_text = self.text2sep_kata(frontend_text)
        # print("sep_text: ", sep_text)
        # print("sep_kata: ", sep_kata)
        # print("frontend_text: ", frontend_text)
        # sep_phonemes: 各単語ごとの音素のリストのリスト
        sep_phonemes = handle_long_word([kata2phoneme_list(i) for i in sep_kata])
        # print("sep_phonemes: ", sep_phonemes)

        pron_text = [x["pron"].strip().replace("’", "") for x in frontend_text]
        # pdb.set_trace()
        prosodys = pyopenjtalk.make_label(frontend_text)
        prosodys = frontend2phoneme(prosodys, drop_unvoiced_vowels=True)
        # print("prosodys: ", ' '.join(prosodys))
        # print("pron_text: ", pron_text)
        normed_text = [x["string"].strip() for x in frontend_text]
        # punctuationがすべて消えた、音素とアクセントのタプルのリスト
        phone_tone_list_wo_punct = g2phone_tone_wo_punct(prosodys)
        # print("phone_tone_list_wo_punct: ", phone_tone_list_wo_punct)

        # phone_w_punct: sep_phonemesを結合した、punctuationを元のまま保持した音素列
        phone_w_punct: list[str] = []
        w_p_len = []
        for i in sep_phonemes:
            phone_w_punct += i
            w_p_len.append(len(i))
        phone_w_punct = phone_w_punct[:-1]
        # punctuation無しのアクセント情報を使って、punctuationを含めたアクセント情報を作る
        # print("phone_w_punct: ", phone_w_punct)
        # print("phone_tone_list_wo_punct: ", phone_tone_list_wo_punct)
        phone_tone_list = align_tones(phone_w_punct, phone_tone_list_wo_punct)

        jp_item = {}
        jp_p = ""
        jp_t = ""
        # mye rye pye bye nye
        # je she
        # print(phone_tone_list)
        for p, t in phone_tone_list:
            if p in self.ipa_dict:
                curr_p = self.ipa_dict[p]
                jp_p += curr_p
                jp_t += str(t + 6) * len(curr_p)
            elif p in punctuation:
                jp_p += p
                jp_t += "0"
            elif p == "▁":
                jp_p += p
                jp_t += " "
            else:
                print(p, t)
            jp_p += "|"
            jp_t += "0"
        # return phones, tones, w_p_len
        jp_p = jp_p.replace("▁", " ")
        jp_t = jp_t.translate(self.table)
        jp_l = ""
        for t in jp_t:
            if t == " ":
                jp_l += " "
            else:
                jp_l += "2"
        # print(jp_p)
        # print(jp_t)
        # print(jp_l)
        # print(len(jp_p_len), sum(w_p_len),  len(jp_p), sum(jp_p_len))
        assert len(jp_p) == len(jp_t) and len(jp_p) == len(jp_l)

        jp_item["jp_p"] = jp_p.replace("| |", "|").rstrip("|")
        jp_item["jp_t"] = jp_t
        jp_item["jp_l"] = jp_l
        jp_item["jp_normed_text"] = " ".join(normed_text)
        jp_item["jp_pron_text"] = " ".join(pron_text)
        # jp_item['jp_ruoma'] = sep_phonemes
        # print(len(normed_text), len(sep_phonemes))
        # print(normed_text)
        return jp_item


jpc = JapanesePhoneConverter()


def japanese_to_ipa(text, text_tokenizer):
    # phonemes = text_tokenizer(text)
    if type(text) == str:
        return jpc.getSentencePhone(text)["jp_p"]
    else:
        result_ph = []
        for t in text:
            result_ph.append(jpc.getSentencePhone(t)["jp_p"])
        return result_ph
