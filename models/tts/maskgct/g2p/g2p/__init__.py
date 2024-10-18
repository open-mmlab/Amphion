# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.tts.maskgct.g2p.g2p import cleaners
from tokenizers import Tokenizer
from models.tts.maskgct.g2p.g2p.text_tokenizers import TextTokenizer
import LangSegment
import json
import re


class PhonemeBpeTokenizer:

    def __init__(self, vacab_path="./models/tts/maskgct/g2p/g2p/vocab.json"):
        self.lang2backend = {
            "zh": "cmn",
            "ja": "ja",
            "en": "en-us",
            "fr": "fr-fr",
            "ko": "ko",
            "de": "de",
        }
        self.text_tokenizers = {}
        self.int_text_tokenizers()

        with open(vacab_path, "r") as f:
            json_data = f.read()
        data = json.loads(json_data)
        self.vocab = data["vocab"]
        LangSegment.setfilters(["en", "zh", "ja", "ko", "fr", "de"])

    def int_text_tokenizers(self):
        for key, value in self.lang2backend.items():
            self.text_tokenizers[key] = TextTokenizer(language=value)

    def tokenize(self, text, sentence, language):

        # 1. convert text to phoneme
        phonemes = []
        if language == "auto":
            seglist = LangSegment.getTexts(text)
            tmp_ph = []
            for seg in seglist:
                tmp_ph.append(
                    self._clean_text(
                        seg["text"], sentence, seg["lang"], ["cjekfd_cleaners"]
                    )
                )
            phonemes = "|_|".join(tmp_ph)
        else:
            phonemes = self._clean_text(text, sentence, language, ["cjekfd_cleaners"])
        # print('clean text: ', phonemes)

        # 2. tokenize phonemes
        phoneme_tokens = self.phoneme2token(phonemes)
        # print('encode: ', phoneme_tokens)

        # # 3. decode tokens [optional]
        # decoded_text = self.tokenizer.decode(phoneme_tokens)
        # print('decoded: ', decoded_text)

        return phonemes, phoneme_tokens

    def _clean_text(self, text, sentence, language, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text, sentence, language, self.text_tokenizers)
        return text

    def phoneme2token(self, phonemes):
        tokens = []
        if isinstance(phonemes, list):
            for phone in phonemes:
                phone = phone.split("\t")[0]
                phonemes_split = phone.split("|")
                tokens.append(
                    [self.vocab[p] for p in phonemes_split if p in self.vocab]
                )
        else:
            phonemes = phonemes.split("\t")[0]
            phonemes_split = phonemes.split("|")
            tokens = [self.vocab[p] for p in phonemes_split if p in self.vocab]
        return tokens
