# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from utils.g2p_new.mandarin import chinese_to_ipa


def cjekfd_cleaners(text, language, text_tokenizers):

    if language == "zh":
        return chinese_to_ipa(text, text_tokenizers["zh"])
    else:
        raise Exception("Unknown or Not supported yet language: %s" % language)
        return None
