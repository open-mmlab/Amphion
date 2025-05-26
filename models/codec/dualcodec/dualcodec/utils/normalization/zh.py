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
    Remove punctuation from text 删除所有非中文、数字和标点符号

    Args:
        text (str): Text to be cleaned
    """
    text = re.sub(
        r"[^，。？！《》‘’“”：；（）【】「」『』~〜～—ー、\u4e00-\u9fff\s_,\-\.\?!;:\"\'…\w()\u3040-\u309F\u30A0-\u30FF]",
        "",
        text,
    )
    # text = re.sub(r"\s*([,\.\?!;:\'…])\s*", r"\1", text)
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
    # text = "114.514，2023年8月16日，是个特别的日子。我们和5000名观众一起召开新项目的发表会。会场在tokyo-domu，开始时间是下午3点。我们起价¥2,000，贵宾席¥10,000。如果有兴趣的话，请一定要参加!！详情请查看我们的网站:https://example.com。谢谢"
    text = '问界 M9 于" 2023 年 12 月 26 日"上市，定位豪华科技旗舰 SUV，也是目前问界家族定位最高、价格最贵的产品。问界 M9 搭载了华为智能汽车全栈技术解决方案，截至目前，问界 M9 累计大定突破 12 万辆，交付超 7 万辆。'
    print(text)
    print(normalize_zh(text, True))
