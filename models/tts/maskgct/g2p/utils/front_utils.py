#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   front_utils.py
@Time    :   2024/08/22 22:10:29
@Author  :   Bo Jin 
@Version :   1.0
@Contact :   jinbo5650@gmail.com
@Brief   :   放前端数据处理工具
'''

import os



def generate_poly_lexicon(file_path:str):
    """Generate poly char lexicon for Mandarin Chinese."""
    poly_dict = {}

    with open(file_path,"r",encoding="utf-8") as readf:
        txt_list = readf.readlines()
        for txt in txt_list:
            word = txt.strip("\n")
            if word not in poly_dict:
                poly_dict[word] = 1
        readf.close()
    return poly_dict