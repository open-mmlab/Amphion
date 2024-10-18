# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from transformers import BertTokenizer
from torch.utils.data import Dataset
from transformers.models.bert.modeling_bert import *
import torch.nn.functional as F
from onnxruntime import InferenceSession, GraphOptimizationLevel, SessionOptions


class PolyDataset(Dataset):
    def __init__(self, words, labels, word_pad_idx=0, label_pad_idx=-1):
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

    def preprocess(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples:
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        data = []
        labels = []
        sentences = []
        # tokenize
        for line in origin_sentences:
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = []
            word_lens = []
            for token in line:
                words.append(token)
                word_lens.append(1)
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append(((words, token_start_idxs), 0))
        ###
        for tag in origin_labels:
            labels.append(tag)

        for sentence, label in zip(sentences, labels):
            data.append((sentence, label))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):

        sentences = [x[0][0] for x in batch]
        ori_sents = [x[0][1] for x in batch]
        labels = [x[1] for x in batch]
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0
        batch_data = np.ones((batch_len, max_len))
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        batch_pmasks = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]
            batch_pmasks[j][:cur_tags_len] = [
                1 if item > 0 else 0 for item in labels[j]
            ]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_pmasks = torch.tensor(batch_pmasks, dtype=torch.long)
        return [batch_data, batch_label_starts, batch_labels, batch_pmasks, ori_sents]


class BertPolyPredict:
    def __init__(self, bert_model, jsonr_file, json_file):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        with open(jsonr_file, "r", encoding="utf8") as fp:
            self.pron_dict = json.load(fp)
        with open(json_file, "r", encoding="utf8") as fp:
            self.pron_dict_id_2_pinyin = json.load(fp)
        self.num_polyphone = len(self.pron_dict)
        self.device = "cpu"
        self.polydataset = PolyDataset
        options = SessionOptions()  # initialize session options
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        print(os.path.join(bert_model, "poly_bert_model.onnx"))
        self.session = InferenceSession(
            os.path.join(bert_model, "poly_bert_model.onnx"),
            sess_options=options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],  # CPUExecutionProvider #CUDAExecutionProvider
        )
        # self.session.set_providers(['CUDAExecutionProvider', "CPUExecutionProvider"], [ {'device_id': 0}])

        # disable session.run() fallback mechanism, it prevents for a reset of the execution provider
        self.session.disable_fallback()

    def predict_process(self, txt_list):
        word_test, label_test, texts_test = self.get_examples_po(txt_list)
        data = self.polydataset(word_test, label_test)
        predict_loader = DataLoader(
            data, batch_size=1, shuffle=False, collate_fn=data.collate_fn
        )
        pred_tags = self.predict_onnx(predict_loader)
        return pred_tags

    def predict_onnx(self, dev_loader):
        pred_tags = []
        with torch.no_grad():
            for idx, batch_samples in enumerate(dev_loader):
                # [batch_data, batch_label_starts, batch_labels, batch_pmasks, ori_sents]
                batch_data, batch_label_starts, batch_labels, batch_pmasks, _ = (
                    batch_samples
                )
                # shift tensors to GPU if available
                batch_data = batch_data.to(self.device)
                batch_label_starts = batch_label_starts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_pmasks = batch_pmasks.to(self.device)
                batch_data = np.asarray(batch_data, dtype=np.int32)
                batch_pmasks = np.asarray(batch_pmasks, dtype=np.int32)
                # batch_output = self.session.run(output_names=['outputs'], input_feed={"input_ids":batch_data, "input_pmasks": batch_pmasks})[0][0]
                batch_output = self.session.run(
                    output_names=["outputs"], input_feed={"input_ids": batch_data}
                )[0]
                label_masks = batch_pmasks == 1
                batch_labels = batch_labels.to("cpu").numpy()
                for i, indices in enumerate(np.argmax(batch_output, axis=2)):
                    for j, idx in enumerate(indices):
                        if label_masks[i][j]:
                            # pred_tag.append(idx)
                            pred_tags.append(self.pron_dict_id_2_pinyin[str(idx + 1)])
        return pred_tags

    def get_examples_po(self, text_list):

        word_list = []
        label_list = []
        sentence_list = []
        id = 0
        for line in [text_list]:
            sentence = line[0]
            words = []
            tokens = line[0]
            index = line[-1]
            front = index
            back = len(tokens) - index - 1
            labels = [0] * front + [1] + [0] * back
            words = ["[CLS]"] + [item for item in sentence]
            words = self.tokenizer.convert_tokens_to_ids(words)
            word_list.append(words)
            label_list.append(labels)
            sentence_list.append(sentence)

            id += 1
            # mask_list.append(masks)
            assert len(labels) + 1 == len(words), print(
                (
                    poly,
                    sentence,
                    words,
                    labels,
                    sentence,
                    len(sentence),
                    len(words),
                    len(labels),
                )
            )
            assert len(labels) + 1 == len(
                words
            ), "Number of labels does not match number of words"
            assert len(labels) == len(
                sentence
            ), "Number of labels does not match number of sentences"
            assert len(word_list) == len(
                label_list
            ), "Number of label sentences does not match number of word sentences"
        return word_list, label_list, text_list
