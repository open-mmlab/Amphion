# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def slice_segments(x, ids_str, segment_size=200):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_ids_segments(lengths, segment_size=200):
    b = lengths.shape[0]
    ids_str_max = lengths - segment_size
    ids_str = (torch.rand([b]).to(device=lengths.device) * ids_str_max).to(
        dtype=torch.long
    )
    return ids_str


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1
