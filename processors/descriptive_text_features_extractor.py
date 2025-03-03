# Copyright (c) 2023 Amphion.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TODO:

This module aims to be an entrance that integrates all the "descriptive text" features extraction functions.

The common descriptive text features include:
1. Global semantic guidance features that extracted some pretrained text models like T5. It can be adopted to TTA, TTM, etc.

Note:
All the features extraction are designed to utilize GPU to the maximum extent, which can ease the on-the-fly extraction for large-scale dataset.

"""
