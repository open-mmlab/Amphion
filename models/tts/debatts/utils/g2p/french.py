"""https://github.com/bootphon/phonemizer"""

import re
from phonemizer import phonemize
from phonemizer.separator import Separator

# List of (regular expression, replacement) pairs for abbreviations in french:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("M", "monsieur"),
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
        ("N.B", "nota bene"),
        ("M", "monsieur"),
        ("p.c.q", "parce que"),
        ("Pr", "professeur"),
        ("qqch", "quelque chose"),
        ("rdv", "rendez-vous"),
        ("max", "maximum"),
        ("min", "minimum"),
        ("no", "numéro"),
        ("adr", "adresse"),
        ("dr", "docteur"),
        ("st", "saint"),
        ("co", "companie"),
        ("jr", "junior"),
        ("sgt", "sergent"),
        ("capt", "capitain"),
        ("col", "colonel"),
        ("av", "avenue"),
        ("av. J.-C", "avant Jésus-Christ"),
        ("apr. J.-C", "après Jésus-Christ"),
        ("art", "article"),
        ("boul", "boulevard"),
        ("c.-à-d", "c’est-à-dire"),
        ("etc", "et cetera"),
        ("ex", "exemple"),
        ("excl", "exclusivement"),
        ("boul", "boulevard"),
    ]
] + [
    (re.compile("\\b%s" % x[0]), x[1])
    for x in [
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
    ]
]

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
        ("j", "jˈ"),  # To avoid incorrect connect
        ("n", "ˈn"),  # To avoid incorrect connect
        ("w", "wˈ"),  # To avoid incorrect connect
        ("ã", "a~"),
        ("ɑ̃", "ɑ~"),
        ("ɔ̃", "ɔ~"),
        ("ɛ̃", "ɛ~"),
        ("œ̃", "œ~"),
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
    text = text.replace("&", " et ")
    return text


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text


def text_normalize(text):
    text = expand_abbreviations(text)
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


def french_to_ipa(text):
    text = text_normalize(text)

    ipa = phonemize(
        text.strip(),
        language="fr-fr",
        backend="espeak",
        separator=Separator(phone=None, word=" ", syllable="|"),
        strip=True,
        preserve_punctuation=True,
        njobs=4,
    )

    # remove "(en)" and "(fr)" tag
    ipa = ipa.replace("(en)", "").replace("(fr)", "")

    ipa = special_map(ipa)

    return ipa
