# Copyright (c) 2023 Amphion.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""

This module aims to be an entrance that integrates all the "audio" features extraction functions.

The common audio features include:
1. Acoustic features such as Mel Spectrogram, F0, Energy, etc.
2. Content features such as phonetic posteriorgrams (PPG) and bottleneck features (BNF) from pretrained

Note: 
All the features extraction are designed to utilize GPU to the maximum extent, which can ease the on-the-fly extraction for large-scale dataset.

"""


class AudioFeaturesExtractor:
    def __init__(self, cfg, wav=None, sr=None):
        """
        Args:
            cfg: Amphion config that would be used to specify the processing parameters
            wav (Tensor, optional): The waveform extracted from. During the on-the-fly extraction, it is usually as the batch input. Defaults to None.
            sr (int, optional): The sampling rate of the waveform. Defaults to None.
        """
        self.cfg = cfg
