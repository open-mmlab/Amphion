"""from https://github.com/Plachtaa/VALL-E-X/g2p"""

import re
from unidecode import unidecode

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]

# List of (romaji, ipa2) pairs for marks:
_romaji_to_ipa2 = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("u", "ɯ"),
        ("ʧ", "tʃ"),
        ("j", "dʑ"),
        ("y", "j"),
        ("ni", "n^i"),
        ("nj", "n^"),
        ("hi", "çi"),
        ("hj", "ç"),
        ("f", "ɸ"),
        ("I", "i*"),
        ("U", "ɯ*"),
        ("r", "ɾ"),
    ]
]

# List of (consonant, sokuon) pairs:
_real_sokuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"Q([↑↓]*[kg])", r"k#\1"),
        (r"Q([↑↓]*[tdjʧ])", r"t#\1"),
        (r"Q([↑↓]*[sʃ])", r"s\1"),
        (r"Q([↑↓]*[pb])", r"p#\1"),
    ]
]

# List of (consonant, hatsuon) pairs:
_real_hatsuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"N([↑↓]*[pbm])", r"m\1"),
        (r"N([↑↓]*[ʧʥj])", r"n^\1"),
        (r"N([↑↓]*[tdn])", r"n\1"),
        (r"N([↑↓]*[kg])", r"ŋ\1"),
    ]
]


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def japanese_to_romaji_with_accent(text):
    """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
    import pyopenjtalk

    text = symbols_to_japanese(text)
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = ""
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if text != "":
                text += " "
            labels = pyopenjtalk.extract_fullcontext(sentence)
            for n, label in enumerate(labels):
                phoneme = re.search(r"\-([^\+]*)\+", label).group(1)
                if phoneme not in ["sil", "pau"]:
                    text += (
                        phoneme.replace("ch", "ʧ").replace("sh", "ʃ").replace("cl", "Q")
                    )
                else:
                    continue
                # n_moras = int(re.search(r'/F:(\d+)_', label).group(1))
                a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
                a2 = int(re.search(r"\+(\d+)\+", label).group(1))
                a3 = int(re.search(r"\+(\d+)/", label).group(1))
                if re.search(r"\-([^\+]*)\+", labels[n + 1]).group(1) in ["sil", "pau"]:
                    a2_next = -1
                else:
                    a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))
                # Accent phrase boundary
                if a3 == 1 and a2_next == 1:
                    text += " "
                # Falling
                elif a1 == 0 and a2_next == a2 + 1:
                    text += "↓"
                # Rising
                elif a2 == 1 and a2_next == 2:
                    text += "↑"
        if i < len(marks):
            text += unidecode(marks[i]).replace(" ", "")
    return text


def get_real_sokuon(text):
    for regex, replacement in _real_sokuon:
        text = re.sub(regex, replacement, text)
    return text


def get_real_hatsuon(text):
    for regex, replacement in _real_hatsuon:
        text = re.sub(regex, replacement, text)
    return text


def japanese_to_ipa(text):
    text = japanese_to_romaji_with_accent(text).replace("...", "…")
    text = get_real_sokuon(text)
    text = get_real_hatsuon(text)
    for regex, replacement in _romaji_to_ipa2:
        text = re.sub(regex, replacement, text)
    return text
