"""https://github.com/bootphon/phonemizer"""

import re

# from g2pkk import G2p
# from jamo import hangul_to_jamo

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

# List of (jamo, ipa) pairs: (need to update)
_jamo_to_ipa = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("ㅏ", "ɐ"),
        ("ㅑ", "jɐ"),
        ("ㅓ", "ʌ"),
        ("ㅕ", "jʌ"),
        ("ㅗ", "o"),
        ("ㅛ", "jo"),
        ("ᅮ", "u"),
        ("ㅠ", "ju"),
        ("ᅳ", "ɯ"),
        ("ㅣ", "i"),
        ("ㅔ", "e"),
        ("ㅐ", "ɛ"),
        ("ㅖ", "je"),
        ("ㅒ", "jɛ"),  # lost
        ("ㅚ", "we"),
        ("ㅟ", "wi"),
        ("ㅢ", "ɯj"),
        ("ㅘ", "wɐ"),
        ("ㅙ", "wɛ"),  # lost
        ("ㅝ", "wʌ"),
        ("ㅞ", "wɛ"),  # lost
        ("ㄱ", "q"),  # 'ɡ' or 'k'
        ("ㄴ", "n"),
        ("ㄷ", "t"),  # d
        ("ㄹ", "ɫ"),  # 'ᄅ' is 'r', 'ᆯ' is 'ɫ'
        ("ㅁ", "m"),
        ("ㅂ", "p"),
        ("ㅅ", "s"),  # 'ᄉ'is 't', 'ᆺ'is 's'
        ("ㅇ", "ŋ"),  # 'ᄋ' is None, 'ᆼ' is 'ŋ'
        ("ㅈ", "tɕ"),
        ("ㅊ", "tɕʰ"),  # tʃh
        ("ㅋ", "kʰ"),  # kh
        ("ㅌ", "tʰ"),  # th
        ("ㅍ", "pʰ"),  # ph
        ("ㅎ", "h"),
        ("ㄲ", "k*"),  # q
        ("ㄸ", "t*"),  # t
        ("ㅃ", "p*"),  # p
        ("ㅆ", "s*"),  # 'ᄊ' is 's', 'ᆻ' is 't'
        ("ㅉ", "tɕ*"),  # tɕ ?
    ]
]

_special_map = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("ʃ", "ɕ"),
        ("tɕh", "tɕʰ"),
        ("kh", "kʰ"),
        ("th", "tʰ"),
        ("ph", "pʰ"),
    ]
]


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


# Convert jamo to IPA
def jamo_to_ipa(text):
    res = ""
    for t in text:
        for regex, replacement in _jamo_to_ipa:
            t = re.sub(regex, replacement, t)
        res += t
    return res


# special map
def special_map(text):
    for regex, replacement in _special_map:
        text = re.sub(regex, replacement, text)
    return text


def korean_to_ipa(text):
    text = normalize(text)

    # espeak-ng
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    ipa = phonemize(
        text,
        language="ko",
        backend="espeak",
        separator=Separator(phone=None, word=" ", syllable="|"),
        strip=True,
        preserve_punctuation=True,
        njobs=4,
    )
    ipa = special_map(ipa)
    # # hangul charactier
    # g2p = G2p()
    # text = g2p(text)
    # text = list(hangul_to_jamo(text))  # '하늘' --> ['ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆯ']
    # ipa = jamo_to_ipa(text)
    return ipa
