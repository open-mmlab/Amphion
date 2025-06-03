from models.tts.vits.vits import (
    slice_segments,
    rand_slice_segments,  # noqa: F401
    get_padding,  # noqa: F401
)

import torch


def rand_spec_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str
