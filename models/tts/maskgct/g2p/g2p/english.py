# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from unidecode import unidecode
import inflect

"""
    Text clean time
"""
_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_percent_number_re = re.compile(r"([0-9\.\,]*[0-9]+%)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_fraction_re = re.compile(r"([0-9]+)/([0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
        ("etc", "et cetera"),
        ("btw", "by the way"),
    ]
]

_special_map = [
    ("t|ɹ", "tɹ"),
    ("d|ɹ", "dɹ"),
    ("t|s", "ts"),
    ("d|z", "dz"),
    ("ɪ|ɹ", "ɪɹ"),
    ("ɐ", "ɚ"),
    ("ᵻ", "ɪ"),
    ("əl", "l"),
    ("x", "k"),
    ("ɬ", "l"),
    ("ʔ", "t"),
    ("n̩", "n"),
    ("oː|ɹ", "oːɹ"),
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_percent(m):
    return m.group(1).replace("%", " percent ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return " " + match + " dollars "  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return " %s %s, %s %s " % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return " %s %s " % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return " %s %s " % (cents, cent_unit)
    else:
        return " zero dollars "


def fraction_to_words(numerator, denominator):
    if numerator == 1 and denominator == 2:
        return " one half "
    if numerator == 1 and denominator == 4:
        return " one quarter "
    if denominator == 2:
        return " " + _inflect.number_to_words(numerator) + " halves "
    if denominator == 4:
        return " " + _inflect.number_to_words(numerator) + " quarters "
    return (
        " "
        + _inflect.number_to_words(numerator)
        + " "
        + _inflect.ordinal(_inflect.number_to_words(denominator))
        + " "
    )


def _expand_fraction(m):
    numerator = int(m.group(1))
    denominator = int(m.group(2))
    return fraction_to_words(numerator, denominator)


def _expand_ordinal(m):
    return " " + _inflect.number_to_words(m.group(0)) + " "


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return " two thousand "
        elif num > 2000 and num < 2010:
            return " two thousand " + _inflect.number_to_words(num % 100) + " "
        elif num % 100 == 0:
            return " " + _inflect.number_to_words(num // 100) + " hundred "
        else:
            return (
                " "
                + _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(
                    ", ", " "
                )
                + " "
            )
    else:
        return " " + _inflect.number_to_words(num, andword="") + " "


# Normalize numbers pronunciation
def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_percent_number_re, _expand_percent, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def _english_to_ipa(text):
    # text = unidecode(text).lower()
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    return text


# special map
def special_map(text):
    for regex, replacement in _special_map:
        regex = regex.replace("|", "\|")
        while re.search(r"(^|[_|]){}([_|]|$)".format(regex), text):
            text = re.sub(
                r"(^|[_|]){}([_|]|$)".format(regex), r"\1{}\2".format(replacement), text
            )
    # text = re.sub(r'([,.!?])', r'|\1', text)
    return text


# Add some special operation
def english_to_ipa(text, text_tokenizer):
    if type(text) == str:
        text = _english_to_ipa(text)
    else:
        text = [_english_to_ipa(t) for t in text]
    phonemes = text_tokenizer(text)
    if phonemes[-1] in "p⁼ʰmftnlkxʃs`ɹaoəɛɪeɑʊŋiuɥwæjː":
        phonemes += "|_"
    if type(text) == str:
        return special_map(phonemes)
    else:
        result_ph = []
        for phone in phonemes:
            result_ph.append(special_map(phone))
        return result_ph
