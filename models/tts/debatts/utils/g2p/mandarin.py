"""from https://github.com/Plachtaa/VALL-E-X/g2p"""

import re
import jieba
import cn2an

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

# List of (bopomofo, romaji) pairs:
_bopomofo_to_romaji = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("ㄅㄛ", "p⁼wo"),
        ("ㄆㄛ", "pʰwo"),
        ("ㄇㄛ", "mwo"),
        ("ㄈㄛ", "fwo"),
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
        ("ㄏ", "h"),
        ("ㄐ", "ʧ⁼"),
        ("ㄑ", "ʧʰ"),
        ("ㄒ", "ʃ"),
        ("ㄓ", "ʦ`⁼"),
        ("ㄔ", "ʦ`ʰ"),
        ("ㄕ", "s`"),
        ("ㄖ", "ɹ`"),
        ("ㄗ", "ʦ⁼"),
        ("ㄘ", "ʦʰ"),
        ("ㄙ", "s"),
        ("ㄚ", "a"),
        ("ㄛ", "o"),
        ("ㄜ", "ə"),
        ("ㄝ", "e"),
        ("ㄞ", "ai"),
        ("ㄟ", "ei"),
        ("ㄠ", "au"),
        ("ㄡ", "ou"),
        ("ㄧㄢ", "yeNN"),
        ("ㄢ", "aNN"),
        ("ㄧㄣ", "iNN"),
        ("ㄣ", "əNN"),
        ("ㄤ", "aNg"),
        ("ㄧㄥ", "iNg"),
        ("ㄨㄥ", "uNg"),
        ("ㄩㄥ", "yuNg"),
        ("ㄥ", "əNg"),
        ("ㄦ", "əɻ"),
        ("ㄧ", "i"),
        ("ㄨ", "u"),
        ("ㄩ", "ɥ"),
        ("ˉ", "→"),
        ("ˊ", "↑"),
        ("ˇ", "↓↑"),
        ("ˋ", "↓"),
        ("˙", ""),
        ("，", ","),
        ("。", "."),
        ("！", "!"),
        ("？", "?"),
        ("—", "-"),
    ]
]

# List of (romaji, ipa) pairs:
_romaji_to_ipa = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("ʃy", "ʃ"),
        ("ʧʰy", "ʧʰ"),
        ("ʧ⁼y", "ʧ⁼"),
        ("NN", "n"),
        ("Ng", "ŋ"),
        ("y", "j"),
        ("h", "x"),
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
        ("ㄧㄢ", "jɛn"),
        ("ㄩㄢ", "ɥæn"),
        ("ㄧㄣ", "in"),
        ("ㄩㄣ", "ɥn"),
        ("ㄧㄥ", "iŋ"),
        ("ㄨㄥ", "ʊŋ"),
        ("ㄩㄥ", "jʊŋ"),
        # Add
        ("ㄧㄚ", "ia"),
        ("ㄧㄝ", "iɛ"),
        ("ㄧㄠ", "iɑʊ"),
        ("ㄧㄡ", "ioʊ"),
        ("ㄧㄤ", "iɑŋ"),
        ("ㄨㄚ", "ua"),
        ("ㄨㄛ", "uo"),
        ("ㄨㄞ", "uaɪ"),
        ("ㄨㄟ", "ueɪ"),
        ("ㄨㄢ", "uan"),
        ("ㄨㄣ", "uən"),
        ("ㄨㄤ", "uɑŋ"),
        ("ㄩㄝ", "ɥɛ"),
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
        ("ㄙ", "s"),
        ("ㄚ", "a"),
        ("ㄛ", "o"),
        ("ㄜ", "ə"),
        ("ㄝ", "ɛ"),
        ("ㄞ", "aɪ"),
        ("ㄟ", "eɪ"),
        ("ㄠ", "ɑʊ"),
        ("ㄡ", "oʊ"),
        ("ㄢ", "an"),
        ("ㄣ", "ən"),
        ("ㄤ", "ɑŋ"),
        ("ㄥ", "əŋ"),
        ("ㄦ", "əɻ"),
        ("ㄧ", "i"),
        ("ㄨ", "u"),
        ("ㄩ", "ɥ"),
        ("ˉ", "→"),
        ("ˊ", "↑"),
        ("ˇ", "↓↑"),
        ("ˋ", "↓"),
        ("˙", ""),
        ("，", ","),
        ("。", "."),
        ("！", "!"),
        ("？", "?"),
        ("—", "-"),
    ]
]


# Convert numbers to Chinese pronunciation
def number_to_chinese(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text


# Word Segmentation, and convert Chinese pronunciation to pinyin (bopomofo)
def chinese_to_bopomofo(text):
    from pypinyin import lazy_pinyin, BOPOMOFO

    text = text.replace("、", "，").replace("；", "，").replace("：", "，")
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
            text += " "
        text += "".join(bopomofos)
    return text


# Convert latin pronunciation to pinyin (bopomofo)
def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text


# Convert pinyin (bopomofo) to Romaji (not used)
def bopomofo_to_romaji(text):
    for regex, replacement in _bopomofo_to_romaji:
        text = re.sub(regex, replacement, text)
    return text


# Convert pinyin (bopomofo) to IPA
def bopomofo_to_ipa(text):
    for regex, replacement in _bopomofo_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


# Convert Chinese to Romaji (not used)
def chinese_to_romaji(text):
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_romaji(text)
    text = re.sub("i([aoe])", r"y\1", text)
    text = re.sub("u([aoəe])", r"w\1", text)
    text = re.sub("([ʦsɹ]`[⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ`\2", text).replace("ɻ", "ɹ`")
    text = re.sub("([ʦs][⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ\2", text)
    return text


# Convert Chinese to IPA
def chinese_to_ipa(text):
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_ipa(text)
    text = re.sub("i([aoe])", r"j\1", text)
    text = re.sub("u([aoəe])", r"w\1", text)
    text = re.sub("([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ`\2", text).replace("ɻ", "ɹ`")
    text = re.sub("([s][⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ\2", text)
    return text
