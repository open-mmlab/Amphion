import sys

f_num_frames = sys.argv[1]
f_utt2spk = sys.argv[2]
threshold = int(sys.argv[3])

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


for spk in spk2utt.keys():
    utts = spk2utt[spk]
    prompts = list(sorted(filter(lambda x:utt2num_frames[x]>threshold, utts), key=lambda x: utt2num_frames[x]))
    if len(prompts) == 0:
        continue
    prompt = prompts[0]
    for utt in utts:
        print(f"{utt} {prompt}")
