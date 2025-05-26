# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset, IterableDataset

path = "DE/*.tar"  # only for testing. please use full data


class EmiliaDataset(IterableDataset):
    def __init__(self, is_debug=True):
        if is_debug:
            self.dataset = load_dataset(
                "amphion/Emilia-Dataset",
                data_files={"de": "DE/*.tar"},
                split="de",
                streaming=True,
            )
        else:
            self.dataset = load_dataset("amphion/Emilia-Dataset", streaming=True)
        # self.dataset = self.dataset.map(lambda x: x, remove_columns=["text", "text_id"])
        # self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        # self.dataset = self.dataset.train_test_split(test_size=0.1)
        # self.dataset = self.dataset["train"]

    def __iter__(self):
        for example in self.dataset:
            yield example

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = EmiliaDataset()
    for example in dataset:
        print(example)
        # (Pdb) example['json']
        # {'dnsmos': 3.4671, 'duration': 17.623, 'id': 'DE_B00000_S00000_W000000', 'language': 'de', 'speaker': 'DE_B00000_S00000', 'text': ' Herzlich Willkommen zu den DSA Nachrichten in 21 Minuten, präsentiert von Hintern im Auge Deiner wöchentlichen Nachrichtensendung rund um DSA, die schwarze Katze und Aventuria. Die heutigen Themen sind ein Sonnenküste Update, die PDFs der zehnten Welle des Collectors Club sind erschienen', 'wav': 'DE_B00000/DE_B00000_S00000/mp3/DE_B00000_S00000_W000000.mp3'}
        # (Pdb) example['mp3']
        # {'path': 'DE_B00000_S00000_W000000.mp3', 'array': array([-1.80987281e-05, -2.03079671e-05, -2.45754873e-05, ...,
        #         1.24948844e-03,  8.32957274e-04,  2.67164229e-04]), 'sampling_rate': 24000}

        # example['mp3']['array'].shape: [T,]
        break
    # dataloader with distributed sampler
    import torch
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
