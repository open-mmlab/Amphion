#!/usr/bin/env python3

from kaldiio import ReadHelper, WriteHelper
import numpy as np
import sys

dict_path = sys.argv[1]
label_rspecifier = sys.argv[2]
vqidx_wspecifier = sys.argv[3]

label2vqidx = {}
with open(dict_path, 'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        label = int(line[0])
        vqidx = list(map(int, line[1:]))
        label2vqidx[label] = vqidx

label_reader = ReadHelper(label_rspecifier)
vqidx_writer = WriteHelper(vqidx_wspecifier)

for utt_id, feat in label_reader:
    feat = feat.astype(np.int64)[:, 0].tolist()
    vqs = []
    for label in feat:
        vqs.append(label2vqidx[label])
    vqidx_writer[utt_id] = np.asarray(vqs).astype(np.float32)

label_reader.close()
vqidx_writer.close()
        
