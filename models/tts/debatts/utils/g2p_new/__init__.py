# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from utils.g2p_new import cleaners
from tokenizers import Tokenizer
from utils.g2p_new.text_tokenizers import TextTokenizer
import json
import re


class PhonemeBpeTokenizer:

    def __init__(self, vacab_path="./utils/g2p_new/vacab.json"):
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
        vacab_path = "./g2p_new/vacab.json"
        with open(vacab_path, "rb") as f:
            json_data = f.read()
        data = json.loads(json_data)
        self.vocab = data["vocab"]

    def int_text_tokenizers(self):
        for key, value in self.lang2backend.items():
            self.text_tokenizers[key] = TextTokenizer(language=value)

    def tokenize(self, text, language):

        # 1. convert text to phoneme
        phonemes = self._clean_text(text, language, ["cjekfd_cleaners"])
        # print('clean text: ', phonemes)

        # 2. tokenize phonemes
        phoneme_tokens = self.phoneme2token(phonemes)
        # print('encode: ', phoneme_tokens)

        # # 3. decode tokens [optional]
        # decoded_text = self.tokenizer.decode(phoneme_tokens)
        # print('decoded: ', decoded_text)

        return phonemes, phoneme_tokens

    def _clean_text(self, text, language, cleaner_names):

        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text, language, self.text_tokenizers)
        return text

    def phoneme2token(self, phonemes):
        # converts phonemes into tokens. In fact, the input phone id is also the first asr audio into text and then converted into token, using the same set of vocab system
        tokens = []
        if isinstance(phonemes, list):
            for phone in phonemes:
                phonemes_split = phone.split("|")
                tokens.append(
                    [self.vocab[p] for p in phonemes_split if p in self.vocab]
                )
        else:
            phonemes_split = phonemes.split("|")
            tokens = [self.vocab[p] for p in phonemes_split if p in self.vocab]
        return tokens
