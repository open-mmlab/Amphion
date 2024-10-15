#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int)
parser.add_argument('--srcdir', type=str)
parser.add_argument('--outdir', type=str)
args = parser.parse_args()

import sys
import os

utt2num_frames = []
with open(os.path.join(args.srcdir, 'utt2num_frames'), 'r') as f:
    for line in f.readlines():
        uttid, num_frames = line.strip().split()
        utt2num_frames.append((uttid, int(num_frames)))

utt2num_frames = sorted(utt2num_frames, key=lambda x: x[1])
num_utts = len(utt2num_frames)
total_num_frames = sum(list(zip(*utt2num_frames))[1])
num_frames_per_part = total_num_frames / args.part

i = 0
for p in range(1, args.part+1):
    part_dir = os.path.join(args.outdir, 'split%d/part%d' % (args.part, p))
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)
    num_frames_current_part = 0
    with open(os.path.join(part_dir, 'utt2num_frames'), 'w') as f:
        while (p == args.part or num_frames_current_part < num_frames_per_part) and i < num_utts:
            uttid, num_frames = utt2num_frames[i]
            f.write('%s %d\n' % (uttid, num_frames))
            num_frames_current_part += num_frames
            i += 1
