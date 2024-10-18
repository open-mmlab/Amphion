# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os


def generate_poly_lexicon(file_path: str):
    """Generate poly char lexicon for Mandarin Chinese."""
    poly_dict = {}

    with open(file_path, "r", encoding="utf-8") as readf:
        txt_list = readf.readlines()
        for txt in txt_list:
            word = txt.strip("\n")
            if word not in poly_dict:
                poly_dict[word] = 1
        readf.close()
    return poly_dict
