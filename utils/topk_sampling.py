# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


# This function is modified from https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits (torch.Tensor): Logits distribution with shape (batch size, vocabulary size).
        top_k (int, optional): Keep only top k tokens with highest probability (top-k filtering).
                               Set to 0 to disable. Defaults to 0.
        top_p (float, optional): Keep the top tokens with a cumulative probability >= top_p (nucleus filtering).
                                 Must be between 0 and 1, inclusive. Defaults to 1.0.
        filter_value (float, optional): The value to assign to filtered logits. Defaults to -float('Inf').
        min_tokens_to_keep (int, optional): Ensure that at least this number of tokens are kept per batch example.
                                            Defaults to 1.

    Returns:
        torch.Tensor: The filtered logits.
    """
    """
        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        # Apply top-k filtering
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k).values[..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        # Apply top-p filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Create a mask to remove tokens with cumulative probability above the top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def topk_sampling(logits, top_k=50, top_p=1.0, temperature=1.0):
    """
    Perform top-k and top-p sampling on logits.

    Args:
        logits (torch.Tensor): The logits to sample from.
        top_k (int, optional): The number of highest probability tokens to keep for top-k filtering.
                               Must be a positive integer. Defaults to 50.
        top_p (float, optional): The cumulative probability threshold for nucleus sampling.
                                 Must be between 0 and 1. Defaults to 1.0.
        temperature (float, optional): The scaling factor to adjust the logits distribution.
                                       Must be strictly positive. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """

    # Adjust logits using temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # Sample from the filtered distribution
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token
