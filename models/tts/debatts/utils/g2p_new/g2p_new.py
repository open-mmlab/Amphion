# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from utils.g2p_new import PhonemeBpeTokenizer
import tqdm

text_tokenizer = PhonemeBpeTokenizer()


def new_g2p(text, language):
    return text_tokenizer.tokenize(text=text, language=language)
