"""https://github.com/bootphon/phonemizer"""

import re
from phonemizer import phonemize
from phonemizer.separator import Separator

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

_special_map = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("ø", "ɸ"),
        ("ː", ":"),
        ("ɜ", "ʒ"),
        ("ɑ̃", "ɑ~"),
        ("j", "jˈ"),  # To avoid incorrect connect
        ("n", "ˈn"),  # To avoid incorrect connect
        ("t", "tˈ"),  # To avoid incorrect connect
        ("ŋ", "ˈŋ"),  # To avoid incorrect connect
        ("ɪ", "ˈɪ"),  # To avoid incorrect connect
    ]
]


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
    text = re.sub(r"([^\.,!\?\-…])$", r"\1.", text)
    return text


# special map
def special_map(text):
    for regex, replacement in _special_map:
        text = re.sub(regex, replacement, text)
    return text


def german_to_ipa(text):
    text = text_normalize(text)

    ipa = phonemize(
        text.strip(),
        language="de",
        backend="espeak",
        separator=Separator(phone=None, word=" ", syllable="|"),
        strip=True,
        preserve_punctuation=True,
        njobs=4,
    )

    # remove "(en)" and "(fr)" tag
    ipa = ipa.replace("(en)", "").replace("(de)", "")

    ipa = special_map(ipa)

    return ipa
