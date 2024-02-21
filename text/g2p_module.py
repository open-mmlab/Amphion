# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
from g2p_en import G2p
from string import punctuation
from typing import Any, Dict, List, Optional, Pattern, Union

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass


# This code is modified from
# https://github.com/lifeiteng/vall-e/blob/9c69096d603ce13174fb5cb025f185e2e9b36ac7/valle/data/tokenizer.py


class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="initials_finals",
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks

    def phonemize(
        self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = get_finals(py[0][:-1], strict=False) + py[0][-1]
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError
            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}")
            )
        return phonemized


class G2PModule:
    """Phonemize Text."""

    # We support espeak to extract IPA (International Phonetic Alphabet), which supports 100 languages,
    # https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        self.separator = separator
        self.backend = self._initialize_backend(
            backend,
            language,
            punctuation_marks,
            preserve_punctuation,
            with_stress,
            tie,
            language_switch,
            words_mismatch,
        )

    def _initialize_backend(
        self,
        backend,
        language,
        punctuation_marks,
        preserve_punctuation,
        with_stress,
        tie,
        language_switch,
        words_mismatch,
    ):
        if backend == "espeak":
            return EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            if language != "cmn":
                raise ValueError(
                    f"{language} is not supported for pypinyin and pypinyin_initials_finals."
                )
            return PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + self.separator.word,
            )
        else:
            raise NotImplementedError(f"{backend}")

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone] + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def phonemization(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        phonemes = [self.to_list(p) for p in phonemized]
        return phonemes

    def g2p_conversion(self, text: str) -> List[str]:
        phonemes = self.phonemization([text.strip()])
        return phonemes[0]


class LexiconModule:
    def __init__(self, lex_path, language="en-us") -> None:
        # todo: check lexicon derivation, merge with G2PModule?
        lexicon = {}
        with open(lex_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        self.lexicon = lexicon
        self.language = language
        self.lang2g2p = {"en-us": G2p()}

    def g2p_conversion(self, text):
        phone = None

        # todo: preprocess with other languages
        if self.language == "en-us":
            phone = self.preprocess_english(text)
        else:
            print("No support to", self.language)
            raise

        return phone

    def preprocess_english(self, text):
        text = text.rstrip(punctuation)

        g2p = self.lang2g2p["en-us"]
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in self.lexicon:
                phones += self.lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))
        phones = "}{".join(phones)
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")

        return phones
