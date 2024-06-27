import speechmetrics
import os
import numpy as np
import sys
from tqdm import tqdm
import soundfile as sf
from pypesq import pesq
# before start, should conda activate vits

synthe_dir = sys.argv[1]
gt_dir = sys.argv[2]  # may contain more samples than synthe_dir

metrics = speechmetrics.load(["stoi", "pesq"])
synthe_pesq = []
synthe_stoi = []
for file in tqdm(os.listdir(synthe_dir)):
    if not file.endswith(".wav"):
        continue
    scores = metrics(os.path.join(synthe_dir, file), os.path.join(gt_dir, file))
    synthe_pesq.append(scores['pesq'].mean())
    synthe_stoi.append(scores['stoi'].mean())

    # ref, sr = sf.read(os.path.join(gt_dir, file))
    # deg, sr = sf.read(os.path.join(synthe_dir, file))
    # score = pesq(ref, deg, sr)
    # print(score)
# exit(0)
print(np.mean(synthe_pesq), np.std(synthe_pesq))
print(np.mean(synthe_stoi), np.std(synthe_stoi))
