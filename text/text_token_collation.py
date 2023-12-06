# Copyright (c) 2023 Amphion.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
import torch
from text.symbol_table import SymbolTable
from text import text_to_sequence


"""
    TextToken: map text to id
"""


# TextTokenCollator is modified from
# https://github.com/lifeiteng/vall-e/blob/9c69096d603ce13174fb5cb025f185e2e9b36ac7/valle/data/collation.py
class TextTokenCollator:
    def __init__(
        self,
        text_tokens: List[str],
        add_eos: bool = True,
        add_bos: bool = True,
        pad_symbol: str = "<pad>",
        bos_symbol: str = "<bos>",
        eos_symbol: str = "<eos>",
    ):
        self.pad_symbol = pad_symbol
        self.add_eos = add_eos
        self.add_bos = add_bos
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        unique_tokens = [pad_symbol]
        if add_bos:
            unique_tokens.append(bos_symbol)
        if add_eos:
            unique_tokens.append(eos_symbol)
        unique_tokens.extend(sorted(text_tokens))

        self.token2idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2token = unique_tokens

    def index(self, tokens_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs, seq_lens = [], []
        for tokens in tokens_list:
            assert all([True if s in self.token2idx else False for s in tokens]) is True
            seq = (
                ([self.bos_symbol] if self.add_bos else [])
                + list(tokens)
                + ([self.eos_symbol] if self.add_eos else [])
            )
            seqs.append(seq)
            seq_lens.append(len(seq))

        max_len = max(seq_lens)
        for k, (seq, seq_len) in enumerate(zip(seqs, seq_lens)):
            seq.extend([self.pad_symbol] * (max_len - seq_len))

        tokens = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )
        tokens_lens = torch.IntTensor(seq_lens)

        return tokens, tokens_lens

    def __call__(self, text):
        tokens_seq = [p for p in text]
        seq = (
            ([self.bos_symbol] if self.add_bos else [])
            + tokens_seq
            + ([self.eos_symbol] if self.add_eos else [])
        )

        token_ids = [self.token2idx[token] for token in seq]
        token_lens = len(tokens_seq) + self.add_eos + self.add_bos

        return token_ids, token_lens


def get_text_token_collater(text_tokens_file: str) -> TextTokenCollator:
    text_tokens_path = Path(text_tokens_file)
    unique_tokens = SymbolTable.from_file(text_tokens_path)
    collater = TextTokenCollator(unique_tokens.symbols, add_bos=True, add_eos=True)
    token2idx = collater.token2idx
    return collater, token2idx


class phoneIDCollation:
    def __init__(self, cfg, dataset=None, symbols_dict_file=None) -> None:
        if cfg.preprocess.phone_extractor != "lexicon":
            ### get text token collator
            if symbols_dict_file is None:
                assert dataset is not None
                symbols_dict_file = os.path.join(
                    cfg.preprocess.processed_dir, dataset, cfg.preprocess.symbols_dict
                )
            self.text_token_colloator, token2idx = get_text_token_collater(
                symbols_dict_file
            )
            # # unique_tokens = SymbolTable.from_file(symbols_dict_path)
            # # text_tokenizer = TextToken(unique_tokens.symbols, add_bos=True, add_eos=True)

            # # update phone symbols dict file with pad_symbol or optional tokens (add_bos and add_eos) in TextTokenCollator
            # phone_symbol_dict = SymbolTable()
            # for s in sorted(list(set(token2idx.keys()))):
            #     phone_symbol_dict.add(s)
            # phone_symbol_dict.to_file(symbols_dict_file)

    def get_phone_id_sequence(self, cfg, phones_seq):
        if cfg.preprocess.phone_extractor == "lexicon":
            phones_seq = " ".join(phones_seq)
            sequence = text_to_sequence(phones_seq, cfg.preprocess.text_cleaners)
        else:
            sequence, seq_len = self.text_token_colloator(phones_seq)
        return sequence
