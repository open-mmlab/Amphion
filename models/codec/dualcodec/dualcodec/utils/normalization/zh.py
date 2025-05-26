# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import cn2an
from .global_punct import normalize_punctuation

# from .text_normalization.cn_tn import TextNorm


def number_to_chinese(text):
    """
    Convert numbers to Chinese pronunciation

    Args:
        text (str): Text to be converted
    """
    return cn2an.transform(text, "an2cn")


def remove_punct(text):
    """
    Remove punctuation from text

    Args:
        text (str): Text to be cleaned
    """
    text = re.sub(
        r"[^，。？！《》‘’“”：；（）【】「」『』~〜～—ー、\u4e00-\u9fff\s_,\-\.\?!;:\"\'…\w()\u3040-\u309F\u30A0-\u30FF]",
        "",
        text,
    )
    return text


def normalize_zh(text, en_punct=False, number_to_chinese=True):
    """
    Normalize Chinese text by lowercasing, number conversion and punctuation removal/normalization

    Args:
        text (str): Text to be normalized
        en_punct (bool): If True, normalize punctuation in the text to English
    """
    text = text.lower()
    if number_to_chinese:
        text = number_to_chinese(text)
    text = remove_punct(text)
    if en_punct:
        text = normalize_punctuation(text)

    return text


if __name__ == "__main__":
    text = "This is a test text with numbers 123 and punctuation."
    print(text)
    print(normalize_zh(text, True))
