#!/usr/bin/env python3

import kaldiio
import numpy as np
import sys

# path to your generated feats.ark file. I.E., the feature matrix. 
feat_path = sys.argv[1]

idx = 0
vq_dict = {}
with open(feat_path, 'r') as f:
    for line in f.readlines():
        uttid, feat = line.strip().split()
        feat = kaldiio.load_mat(feat).astype(np.int64)
        for i in range(len(feat)):
            vq = " ".join(map(str, feat[i].tolist()))
            if not vq in vq_dict:
                vq_dict[vq] = idx
                idx += 1
vq_labels = sorted(list(vq_dict.items()), key=lambda x: x[1])
for vq, label in vq_labels:
    print(label, vq)


