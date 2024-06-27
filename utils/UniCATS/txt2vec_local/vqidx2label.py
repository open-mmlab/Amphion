#!/usr/bin/env python3

from kaldiio import ReadHelper, WriteHelper
import numpy as np
import sys

dict_path = sys.argv[1]
vqidx_rspecifier = sys.argv[2]
label_wspecifier = sys.argv[3]

vqidx2label = {}
with open(dict_path, 'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        label = int(line[0])
        vqidx = " ".join(line[1:])
        vqidx2label[vqidx] = label

vqidx_reader = ReadHelper(vqidx_rspecifier)
label_writer = WriteHelper(label_wspecifier)

for utt_id, feat in vqidx_reader:
    feat = feat.astype(np.int64)
    labels = []
    for i in range(len(feat)):
        vq = " ".join(map(str, feat[i].tolist()))
        labels.append(vqidx2label[vq])
    label_writer[utt_id] = np.expand_dims(np.asarray(labels), axis=-1).astype(np.float32)

vqidx_reader.close()
label_writer.close()
