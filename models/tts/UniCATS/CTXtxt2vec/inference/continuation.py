from tqdm import tqdm
import time
import numpy as np
import logging

import kaldiio
import torch
import argparse
import os
from models.tts.UniCATS.CTXtxt2vec.build_model.utils.io import load_json_config
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.build import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="eval_clean", type=str, help="A data set directory in `data/`. "
                                                                           "This should contain text, duration, feats.scp, utt2prompt files.")
    parser.add_argument("--expdir", default='OUTPUT/Libritts', type=str, help="model training directory")
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()
    eval_set = args.eval_set
    expdir = args.expdir
    device = args.device
    config = load_json_config(f'{expdir}/configs/config.json')
    model = build_model(config).to(device)
    ckpt = torch.load(f"{expdir}/checkpoint/last.pth")
    outdir = f"{expdir}/syn/{eval_set}/"
    model.load_state_dict(ckpt["model"])

    lexicon = {}

    lexicon_file = "data/lang_1phn/train_all_units.txt"
    logging.info(f"Reading {lexicon_file} for valid phones ...")
    with open(lexicon_file, 'r') as f:
        for line in f.readlines():
            txt_token, token_id = line.strip().split()
            lexicon[txt_token] = int(token_id)

    vqid_table = []
    label2vqidx_file = "feats/vqidx/label2vqidx"
    logging.info(f"Reading {label2vqidx_file} for valid VQ indexes")
    with open(label2vqidx_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            label = int(line[0])
            vqid_table.append(torch.tensor(list(map(int, line[1:]))))
    vqid_table = torch.stack(vqid_table, dim=0).to(device)

    utt2text = {}
    text_decode = f"data/{eval_set}/text"
    logging.info(f"Reading {text_decode} for text to be synthesized")
    with open(text_decode) as f:
        for line in f.readlines():
            utt, text = line.strip().split(maxsplit=1)
            utt2text[utt] = text

    utt2dur = {}
    duration_file = f"data/{eval_set}/duration"
    logging.info(f"Reading {duration_file} for duration of prompt")
    with open(duration_file) as f:
        for line in f.readlines():
            utt, duration = line.strip().split(maxsplit=1)
            utt2dur[utt] = list(map(int, duration.split()))

    logging.info(f"Reading data/{eval_set}/feats.scp for VQ index of prompt")
    feats_loader = kaldiio.load_scp(f'data/{eval_set}/feats.scp')

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=os.path.join(os.getcwd(), f"{outdir}/feats")))
    logging.info(f"Writing location: {outdir}/feats.scp")

    with open(f"data/{eval_set}/utt2prompt") as f:
        f_lines = f.readlines()
        logging.info(f"Number of text to be synthesized: {len(f_lines)}")
        logging.info("Decoding starts...")
        with tqdm(f_lines, desc="Decoding") as pbar:
            model.set_generate_type('top0.85r')
            for l in pbar:
                utt, prompt = l.strip().split(maxsplit=1)
                pbar.set_postfix(utt=utt)
                text = utt2text[utt]
                text = torch.LongTensor([lexicon[w] for w in text.split()]).unsqueeze(0).to(device)

                prefix_text = torch.LongTensor([lexicon[w] for w in utt2text[prompt].split()]).unsqueeze(0).to(device)

                duration = torch.LongTensor(utt2dur[utt]).unsqueeze(0).to(device)
                prefix_duration = torch.LongTensor(utt2dur[prompt]).unsqueeze(0).to(device)

                feat = torch.LongTensor(feats_loader[utt][:, -1].copy()).unsqueeze(0).to(device)
                prefix_feat = torch.LongTensor(feats_loader[prompt][:, -1].copy()).unsqueeze(0).to(device)

                out = model.transformer.sample(text, {'text': prefix_text, 'duration': prefix_duration, 'feat': prefix_feat})['content_token'][0]
                out = out[prefix_feat.size(-1):]
                vqid = vqid_table[out].float().cpu().numpy()
                feat_writer[utt] = vqid

    feat_writer.close()
