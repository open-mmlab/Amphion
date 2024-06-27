from kaldiio import ReadHelper, WriteHelper
import sys
from tqdm import tqdm


dim_range = sys.argv[1]
rspecifier = sys.argv[2]
wspecifier = sys.argv[3]


# The Kaldi `select-feats` command provides more powerful dim range parsing functionality,
# like 0,12-24,34-35. But we will only implement xx-yy here.


def parse_dim_range(string):
    # x-y dim: includes both x and y (starting from 0)
    # or a single digit: then only that dimension
    # returns x, y+1
    string = string.strip()
    if "-" in string:
        terms = string.split("-")
        assert len(terms) == 2
        start = int(terms[0])
        end = int(terms[1]) + 1
    else:
        start = int(string)
        end = start + 1
    return start, end

start, end = parse_dim_range(dim_range)
with ReadHelper(rspecifier) as reader, WriteHelper(wspecifier) as writer:
    for uttid, array_out in tqdm(reader, desc="slicing feats.scp"):
        writer(uttid, array_out[:, start:end])
