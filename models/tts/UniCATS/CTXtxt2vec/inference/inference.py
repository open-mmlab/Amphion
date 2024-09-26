from tqdm import tqdm
import kaldiio
import torch

import os
from models.tts.UniCATS.CTXtxt2vec.build_model.utils.io import load_json_config
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.build import build_model

device = "cuda"
config = load_json_config('OUTPUT/Libritts/configs/config.json')
model = build_model(config).to(device)
ckpt = torch.load("OUTPUT/Libritts/checkpoint/last.pth")
model.load_state_dict(ckpt["model"])

lexicon = {}
with open("data/lang_1phn/train_all_units.txt", 'r') as f:
    for line in f.readlines():
        txt_token, token_id = line.strip().split()
        lexicon[txt_token] = int(token_id)

vqid_table = []
with open("feats/vqidx/label2vqidx", 'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        label = int(line[0])
        vqid_table.append(torch.tensor(list(map(int, line[1:]))))
vqid_table = torch.stack(vqid_table, dim=0).to(device)

feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=os.path.join(os.getcwd(), "OUTPUT/Libritts/syn/feats")))
with open("data/eval_all/text") as f:
    for l in tqdm(f.readlines()):
        utt, text = l.strip().split(maxsplit=1)
        text = torch.LongTensor([lexicon[w] for w in text.split()]).unsqueeze(0).to(device)
        out = model.generate('top0.85r', text)['content_token'][0]
        feat_writer[utt] = vqid_table[out].float().cpu().numpy()

feat_writer.close()
