from resemblyzer import preprocess_wav, VoiceEncoder
import warnings
import kaldiio
import numpy as np
import sys
import os

warnings.filterwarnings('ignore') # setting ignore as a parameter
# synthe_dir = "/mnt/lustre/sjtu/home/cpd30/aaai_wavs/seen/v2w"
synthe_dir=sys.argv[1]
data_dir_to_find_prompt = "cpd_kaldi_data/eval_clean"
data_dir_to_find_prompt = sys.argv[2]
data_dir = "cpd_kaldi_data/eval_clean_subset"
data_dir = sys.argv[3]

encoder = VoiceEncoder()

spk2prompt = dict()
with open(f"{data_dir}/spk2prompt", 'r') as fr: 
    lines = fr.readlines()
    for line in lines:
        terms = line.strip().split()
        spk2prompt[terms[0]] = terms[1]

utt2wav = dict()
with open(f"{data_dir_to_find_prompt}/wav.scp", 'r') as fr:  # prompts must be found at a larger set
    for line in fr.readlines():
        terms = line.strip().split()
        utt2wav[terms[0]] = terms[1]

synthe_base_wavs = os.listdir(synthe_dir)
synthe_base_wavs = list(filter(lambda x: x.endswith("wav"), synthe_base_wavs))
synthe_wavs = list(map(lambda x: f"{synthe_dir}/{x}", synthe_base_wavs))
spk2synthe_wavs = dict()
for i in range(len(synthe_base_wavs)):
    spk = synthe_base_wavs[i].split("_")[0] + "_"
    if spk in spk2synthe_wavs:
        spk2synthe_wavs[spk].append(synthe_wavs[i])
    else:
        spk2synthe_wavs[spk] = [synthe_wavs[i]]

spk_wise_score = []
utt_wise_score = []
for spk in spk2synthe_wavs:
    embeds = np.array([encoder.embed_utterance(preprocess_wav(f)) for f in spk2synthe_wavs[spk]])
    embed_prompt_wav = utt2wav[spk2prompt[spk]]
    embed_prompt = encoder.embed_utterance(preprocess_wav(embed_prompt_wav)) # [256]
    res = np.einsum("BN,N -> B", embeds, embed_prompt)
    spk_wise_score.append(res.mean())
    utt_wise_score.extend(res.tolist())
    
print(np.mean(spk_wise_score), np.std(utt_wise_score))

