import numpy as np
import torch
import pandas as pd
from data.filter_data import get_event_list


class Text_Onset_2_Audio_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, args):

        self.captions = list(dataset[args.text_column])
        self.audios = list(dataset[args.audio_column])
        self.onsets = list(dataset[args.onset_column])
        self.indices = list(range(len(self.captions)))

        self.mapper = {}
        for index, audio, caption, onset in zip(
            self.indices, self.audios, self.captions, self.onsets
        ):
            self.mapper[index] = [audio, caption, onset]

        num_examples = args.num_examples
        if num_examples != -1:
            self.captions, self.audios, self.onsets = (
                self.captions[:num_examples],
                self.audios[:num_examples],
                self.onsets[:num_examples],
            )
            self.indices = self.indices[:num_examples]
        self.class2id = {event: idx for idx, event in enumerate(args.event_list)}

    def decode_data(self, line_onset_str):
        # data    { "location":     audio_path,
        #           "captions" :    "event1 n times and event2 n times",
        #           "onset_str":        "event1__onset1-offset1_onset2-offset2--event2__onset1-offset1"}

        line_onset_index = np.zeros((32, 256))
        line_event = []
        for event_onset in line_onset_str.split("--"):
            # event_onset : event1__onset1-offset1_onset2-offset2
            (event, instance) = event_onset.split("__")
            line_event.append(event)
            # instance : onset1-offset1_onset2-offset2
            for start_end in instance.split("_"):
                (start, end) = start_end.split("-")
                start, end = int(float(start) * 250 / 10), int(float(end) * 250 / 10)
                if end > 255:
                    break
                line_onset_index[self.class2id[event], start:end] = 1
        line_event_str = " and ".join(line_event)
        return line_onset_index, line_event_str

    def __len__(self):
        return len(self.captions)

    def get_num_instances(self):
        return len(self.captions)

    def __getitem__(self, index):
        onset_str, filename, idx, caption = (
            self.onsets[index],
            self.audios[index],
            self.indices[index],
            self.captions[index],
        )
        onset, _ = self.decode_data(onset_str)
        # "onset_str":        "event1__onset1-offset1_onset2-offset2--event2__onset1-offset1"
        # assert len(onset_str.split("--")) == 1
        first_class_id = self.class2id[onset_str.split("__")[0]]
        return idx, onset, first_class_id, filename, caption, onset_str

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        batch = []
        for i in dat:
            if i == 1:
                batch.append(
                    torch.tensor(np.array(dat[i].tolist()), dtype=torch.float32)
                )
            elif i == 2:
                batch.append(torch.tensor(dat[i]))
            else:
                batch.append(dat[i].tolist())
        return batch


class Clap_Onset_2_Audio_Dataset(Text_Onset_2_Audio_Dataset):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        import laion_clap
        from laion_clap.clap_module.factory import (
            load_state_dict as clap_load_state_dict,
        )

        self.clap_scorer = laion_clap.CLAP_Module(enable_fusion=False)
        ckpt_path = "miniconda3/envs/py3.10.11/lib/python3.10/site-packages/laion_clap/630k-audioset-best.pt"
        ckpt = clap_load_state_dict(ckpt_path, skip_params=True)
        del_parameter_key = ["text_branch.embeddings.position_ids"]
        ckpt = {"model." + k: v for k, v in ckpt.items() if k not in del_parameter_key}
        self.clap_scorer.load_state_dict(ckpt)

    def __getitem__(self, index):
        onset_str, filename, idx, caption = (
            self.onsets[index],
            self.audios[index],
            self.indices[index],
            self.captions[index],
        )
        onset, event = self.decode_data(onset_str)
        with torch.no_grad():
            clap_embed = self.clap_scorer.get_text_embedding(
                [event, ""], use_tensor=False
            )[0]
        return idx, onset, clap_embed, filename, caption, onset_str

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        batch = []
        for i in dat:
            if i == 1 or i == 2:
                batch.append(
                    torch.tensor(np.array(dat[i].tolist()), dtype=torch.float32)
                )
            else:
                batch.append(dat[i].tolist())
        return batch


if __name__ == "__main__":
    import torch
    from torch.utils.data import Dataset, DataLoader
    import datasets
    import argparse
    import sys

    import models.controllable_dataset as ConDataset
    from data_utils.filter_data import get_event_list

    parser = argparse.ArgumentParser(description=".")
    args = parser.parse_args()
    args.event_list = get_event_list()
    args.train_file = ""

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files={"train": args.train_file})
    train_dataset = Clap_Onset_2_Audio_Dataset(raw_datasets["train"], args)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    for batch in train_dataloader:
        import pdb

        pdb.set_trace()
        idx, onset, event_info, audios, caption, onset_str = batch
