# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import inflect
from .global_punct import normalize_punctuation

# List of known special names
special_names = {"NASA", "UNESCO", "FBI", "USA", "AI"}


def expand_numbers(text):
    """F"""
    p = inflect.engine()

    def replace_number(match):
        return p.number_to_words(match.group(0))

    return re.sub(r"\b\d+\b", replace_number, text)


import re


def preserve_special_names(text):
    """F"""
    # Detect uppercase acronyms with two or more letters
    pattern = r"\b[A-Z]{2,}\b"

    # Replace true acronyms by splitting them into individual uppercase letters
    def replace_special(match):
        word = match.group(0)
        return " ".join(word)  # Split acronyms into individual letters

    # Expand acronyms in the text
    text_with_expanded_acronyms = re.sub(pattern, replace_special, text)

    # Lowercase all other words except already-expanded acronyms
    def lowercase_except_acronyms(match):
        """F"""
        word = match.group(0)
        # Keep expanded acronyms in uppercase
        if all(c.isupper() or c == " " for c in word):
            return word
        else:
            return word.lower()

    # Match words including those with apostrophes
    normalized_text = re.sub(
        r"\b[\w']+\b", lowercase_except_acronyms, text_with_expanded_acronyms
    )

    return normalized_text


def normalize_en(text):
    """F"""
    # Lowercase the entire text
    # text = text.lower()
    text = preserve_special_names(text)

    # Isolate punctuation
    text = normalize_punctuation(text)

    # Remove trailing whitespace
    text = text.rstrip()

    # Expand numbers
    text = expand_numbers(text)

    # Replace single quotes with double quotes after a space
    text = re.sub(r"(?<=\s)'([^']*)'", r'"\1"', text)

    text = text.replace("'", "^")
    # text = text.replace('"', "")
    text = text.replace(";", ",")
    text = text.replace(":", ",")

    # avoid single quote (too rare in data)
    # text = text.replace(" '", ' "')
    # text = text.replace(",'", ', "')
    # text = text.replace("',", '",')
    # text = text.replace("' ,", '",')

    return text


if __name__ == "__main__":
    """F"""
    # Example usage
    text = "It can't believe it's-- Captain USA!~ al-ready 2023! and i According to WHO, COVID-19, the AI GeneraTion in the White House is sick! She （said） \"hello\" (and) ... :waved. taxi to level 210."
    normalized_text = normalize_en(text)
    print(normalized_text)
