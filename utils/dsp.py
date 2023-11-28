# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

# ZERO = 1e-12


def gaussian_normalize_mel_channel(mel, mu, sigma):
    """
    Shift to Standorm Normal Distribution

    Args:
        mel: (n_mels, frame_len)
        mu: (n_mels,), mean value
        sigma: (n_mels,), sd value
    Return:
        Tensor like mel
    """
    mu = np.expand_dims(mu, -1)
    sigma = np.expand_dims(sigma, -1)
    return (mel - mu) / sigma


def de_gaussian_normalize_mel_channel(mel, mu, sigma):
    """

    Args:
        mel: (n_mels, frame_len)
        mu: (n_mels,), mean value
        sigma: (n_mels,), sd value
    Return:
        Tensor like mel
    """
    mu = np.expand_dims(mu, -1)
    sigma = np.expand_dims(sigma, -1)
    return sigma * mel + mu


def decompress(audio_compressed, bits):
    mu = 2**bits - 1
    audio = np.sign(audio_compressed) / mu * ((1 + mu) ** np.abs(audio_compressed) - 1)
    return audio


def compress(audio, bits):
    mu = 2**bits - 1
    audio_compressed = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    return audio_compressed


def label_to_audio(quant, bits):
    classes = 2**bits
    audio = 2 * quant / (classes - 1.0) - 1.0
    return audio


def audio_to_label(audio, bits):
    """Normalized audio data tensor to digit array

    Args:
        audio (tensor): audio data
        bits (int): data bits

    Returns:
        array<int>: digit array of audio data
    """
    classes = 2**bits
    # initialize an increasing array with values from -1 to 1
    bins = np.linspace(-1, 1, classes)
    # change value in audio tensor to digits
    quant = np.digitize(audio, bins) - 1
    return quant


def label_to_onehot(x, bits):
    """Converts a class vector (integers) to binary class matrix.
    Args:
        x: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Returns:
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    classes = 2**bits

    result = torch.zeros((x.shape[0], classes), dtype=torch.float32)
    for i in range(x.shape[0]):
        result[i, x[i]] = 1

    output_shape = x.shape + (classes,)
    output = torch.reshape(result, output_shape)
    return output
