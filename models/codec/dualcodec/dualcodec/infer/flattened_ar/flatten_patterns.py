# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from einops import rearrange

import numpy as np


def offset_codes(semantic_code, offset_sizes):
    """
    Applies layer-specific offsets to each codec layer.

    Args:
        semantic_code (torch.Tensor): Input tensor of shape (batch_size, T, num_codec_layers).
        offset_sizes (list[int]): List of offsets for each codec layer to distinguish them.

    Returns:
        torch.Tensor: Offset-applied tensor of shape (batch_size, T, num_codec_layers).
    """
    # Calculate cumulative offsets for each layer
    cumulative_offsets = np.cumsum(
        [0] + offset_sizes[:-1]
    )  # Start with 0 for the first layer
    # Apply offsets layer by layer
    offsetted_code = []
    for i, offset in enumerate(cumulative_offsets):
        current_layer_code = semantic_code[..., i].clone().detach()  # Extract layer i
        current_layer_code += offset  # Apply the cumulative offset
        offsetted_code.append(current_layer_code)

    # Stack all layers along the codec layer dimension
    offsetted_code = torch.stack(
        offsetted_code, dim=-1
    )  # Shape: (batch_size, T, num_codec_layers)

    return offsetted_code


def deoffset_codes(flattened_codes, offset_sizes):
    """
    De-offsets a flattened tensor by subtracting the codebook size offsets for each codec layer.

    Args:
        flattened_codes (torch.Tensor): The offset and flattened tensor of shape (batch_size, T * num_codec_layers).
        codebook_sizes (list[int]): A list of codebook sizes for each codec layer, used to remove offsets.

    Returns:
        torch.Tensor: The de-offset tensor of shape (batch_size, T, num_codec_layers).
    """
    # Calculate cumulative offsets for each layer
    cumulative_offsets = np.cumsum(
        [0] + offset_sizes[:-1]
    )  # Start with 0 for the first layer

    # Determine dimensions for reshaping
    batch_size, flattened_dim = flattened_codes.shape
    num_codec_layers = len(offset_sizes)
    T = flattened_dim // num_codec_layers

    # Reshape flattened_codes back to (batch_size, T, num_codec_layers)
    reshaped_codes = flattened_codes.view(batch_size, T, num_codec_layers)

    # De-offset each layer by subtracting the respective cumulative offset
    deoffsetted_code = []
    for i, offset in enumerate(cumulative_offsets):
        current_layer_code = reshaped_codes[
            ..., i
        ].clone()  # Clone to avoid in-place operation
        current_layer_code = current_layer_code - offset  # Remove the cumulative offset
        deoffsetted_code.append(current_layer_code)

    # Stack all layers along the codec layer dimension
    deoffsetted_code = torch.stack(
        deoffsetted_code, dim=-1
    )  # Shape: (batch_size, T, num_codec_layers)

    return deoffsetted_code
