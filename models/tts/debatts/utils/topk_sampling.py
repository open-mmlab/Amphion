# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    # pred_token = torch.multinomial(F.softmax(logits, -1), 1) # [BATCH_SIZE, 1]
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
