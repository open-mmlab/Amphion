from torch.utils.data import Dataset
import os
import random
import kaldiio
import torch
import logging


class TTSDataset(Dataset):
    def __init__(self, data_root, lexicon_path, batch_max_tokens, max_seconds=32, min_seconds=5, input_hz: int = 100):
        logging.info(f"Constructing TTSDataset from {data_root}, {lexicon_path}")
        lexicon = {}
        with open(lexicon_path, 'r') as f:
            for line in f.readlines():
                txt_token, token_id = line.strip().split()
                lexicon[txt_token] = int(token_id)

        self.feats = kaldiio.load_scp(os.path.join(data_root, 'feats.scp'))
        logging.info(f"Registered {len(self.feats)} utterance features")

        self.text = {}
        self.duration = {}
        with open(os.path.join(data_root, 'text')) as f:
            for l in f.readlines():
                utt, text = l.strip().split(maxsplit=1)
                self.text[utt] = [lexicon[w] for w in text.split()]
        logging.info(f"Registered {len(self.text)} texts")
        with open(os.path.join(data_root, 'duration')) as f:
            for l in f.readlines():
                utt, duration = l.strip().split(maxsplit=1)
                self.duration[utt] = list(map(int, duration.split()))
        logging.info(f"Registered {len(self.duration)} duration sequences")

        max_frames = int(input_hz * max_seconds)
        min_frames = int(input_hz * min_seconds)
        self.utt2num_frames = {}
        if os.path.exists(os.path.join(data_root, 'utt2num_frames')):
            with open(os.path.join(data_root, 'utt2num_frames')) as f:
                for l in f.readlines():
                    utt, num_frames = l.strip().split(maxsplit=1)
                    num_frames = int(num_frames)
                    if num_frames > max_frames or num_frames < min_frames:
                        continue
                    self.utt2num_frames[utt] = num_frames
        else:
            for utt, feat in self.feats.items():
                num_frames = feat.shape[0]
                if num_frames > max_frames or num_frames < min_frames:
                    continue
                self.utt2num_frames[utt] = num_frames
        logging.info(f"Registered {len(self.utt2num_frames)} utterances within the length range")

        self.utt_ids = sorted(list(set(self.feats.keys()) &
                                   set(self.text.keys()) &
                                   set(self.duration.keys()) &
                                   set(self.utt2num_frames.keys())),
                              key=lambda x: self.utt2num_frames[x])
        logging.info(f"Finally {len(self.utt_ids)} utterances will be used for training")

        self.batches = self.batchfy(batch_max_tokens)

    def batchfy(self, batch_max_tokens):
        batches = []
        current_batch_tokens = 0
        current_batch = []
        for i, utt in enumerate(self.utt_ids):
            num_frames = self.utt2num_frames[utt]
            if len(current_batch) == 0 or current_batch_tokens + num_frames <= batch_max_tokens:
                current_batch.append(utt)
                current_batch_tokens += num_frames
            else:
                batches.append(current_batch)
                current_batch_tokens = num_frames
                current_batch = [utt]
                if i == len(self.utt_ids) - 1:
                    batches.append(current_batch)
        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch = self.batches[index]
        batch_feat = []
        batch_text = []
        batch_duration = []
        for utt_id in batch:
            batch_feat.append(self.feats[utt_id][:, -1])
            batch_text.append(self.text[utt_id])
            batch_duration.append(self.duration[utt_id])
        feat_max_length = max([f.shape[0] for f in batch_feat])
        txt_max_length = max([len(t) for t in batch_text])
        batch_size = len(batch)
        batch_feat_tensor = torch.zeros((batch_size, feat_max_length), dtype=torch.int64)
        batch_text_tensor = torch.zeros((batch_size, txt_max_length), dtype=torch.int64)
        batch_duration_tensor = torch.zeros((batch_size, txt_max_length), dtype=torch.int64)
        batch_feat_len = []
        batch_text_len = []
        for i in range(batch_size):
            feat = batch_feat[i]
            text = batch_text[i]
            duration = batch_duration[i]
            batch_feat_tensor[i, :len(feat)] = torch.tensor(feat).long()
            batch_text_tensor[i, :len(text)] = torch.tensor(text).long()
            batch_duration_tensor[i, :len(duration)] = torch.tensor(duration).long()
            batch_feat_len.append(len(feat))
            batch_text_len.append(len(text))
        batch_feat_len = torch.tensor(batch_feat_len, dtype=torch.int64)
        batch_text_len = torch.tensor(batch_text_len, dtype=torch.int64)

        return {'label': batch_feat_tensor, 'feat_len': batch_feat_len, 'text': batch_text_tensor, 'text_len': batch_text_len,
                'duration': batch_duration_tensor}
