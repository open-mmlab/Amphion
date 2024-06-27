import sys

f_num_frames = sys.argv[1]
f_utt2spk = sys.argv[2]
f_feats_scp = sys.argv[3]
threshold = int(sys.argv[4])

utt2num_frames = {}
with open(f_num_frames) as f:
    for line in f.readlines():
        utt, num_frame = line.strip().split()
        utt2num_frames[utt] = int(num_frame)

spk2utt = {}
with open(f_utt2spk) as f:
    for line in f.readlines():
        utt, spk = line.strip().split()
        if spk2utt.get(spk) is not None:
            spk2utt[spk].append(utt)
        else:
            spk2utt[spk] = [utt]

feats_scp = {}
with open(f_feats_scp) as f:
    for line in f.readlines():
        utt, feat = line.strip().split()
        feats_scp[utt] = feat

for spk in spk2utt.keys():
    utts = spk2utt[spk]
    # filter utts whose num_frames is higher than threshold. Then sort them by num_frames.
    prompts = list(sorted(filter(lambda x: utt2num_frames[x] > threshold, utts), key=lambda x: utt2num_frames[x]))
    if len(prompts) == 0:
        continue
    prompt = prompts[0]  # take the shortest one
    for utt in utts:
        print(f"{utt} {feats_scp[prompt]}")
