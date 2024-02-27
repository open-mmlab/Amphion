# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm
from text.g2p_module import G2PModule, LexiconModule
from text.symbol_table import SymbolTable

"""
    phoneExtractor: extract phone from text
"""


class phoneExtractor:
    def __init__(self, cfg, dataset_name=None, phone_symbol_file=None):
        """
        Args:
            cfg: config
            dataset_name: name of dataset
        """
        self.cfg = cfg

        #  phone symbols dict
        self.phone_symbols = set()

        # phone symbols dict file
        if phone_symbol_file is not None:
            self.phone_symbols_file = phone_symbol_file
        elif dataset_name is not None:
            self.dataset_name = dataset_name
            self.phone_symbols_file = os.path.join(
                cfg.preprocess.processed_dir, dataset_name, cfg.preprocess.symbols_dict
            )

        # initialize g2p module
        if cfg.preprocess.phone_extractor in [
            "espeak",
            "pypinyin",
            "pypinyin_initials_finals",
        ]:
            self.g2p_module = G2PModule(
                backend=cfg.preprocess.phone_extractor, language=cfg.preprocess.language
            )
        elif cfg.preprocess.phone_extractor == "lexicon":
            assert cfg.preprocess.lexicon_path != ""
            self.g2p_module = LexiconModule(cfg.preprocess.lexicon_path)
        else:
            print("No support to", cfg.preprocess.phone_extractor)
            raise

    def extract_phone(self, text):
        """
        Extract phone from text
        Args:

            text:  text of utterance

        Returns:
            phone_symbols: set of phone symbols
            phone_seq: list of phone sequence of each utterance
        """

        if self.cfg.preprocess.phone_extractor in [
            "espeak",
            "pypinyin",
            "pypinyin_initials_finals",
        ]:
            text = text.replace("”", '"').replace("“", '"')
            phone = self.g2p_module.g2p_conversion(text=text)
            self.phone_symbols.update(phone)
            phone_seq = [phn for phn in phone]

        elif self.cfg.preprocess.phone_extractor == "lexicon":
            phone_seq = self.g2p_module.g2p_conversion(text)
            phone = phone_seq
            if not isinstance(phone_seq, list):
                phone_seq = phone_seq.split()

        return phone_seq

    def save_dataset_phone_symbols_to_table(self):
        # load and merge saved phone symbols
        if os.path.exists(self.phone_symbols_file):
            phone_symbol_dict_saved = SymbolTable.from_file(
                self.phone_symbols_file
            )._sym2id.keys()
            self.phone_symbols.update(set(phone_symbol_dict_saved))

        # save phone symbols
        phone_symbol_dict = SymbolTable()
        for s in sorted(list(self.phone_symbols)):
            phone_symbol_dict.add(s)
        phone_symbol_dict.to_file(self.phone_symbols_file)


def extract_utt_phone_sequence(dataset, cfg, metadata):
    """
    Extract phone sequence from text
    Args:
        dataset (str): name of dataset, e.g. opencpop
        cfg: config
        metadata: list of dict, each dict contains "Uid", "Text"

    """

    dataset_name = dataset

    # output path
    out_path = os.path.join(
        cfg.preprocess.processed_dir, dataset_name, cfg.preprocess.phone_dir
    )
    os.makedirs(out_path, exist_ok=True)

    phone_extractor = phoneExtractor(cfg, dataset_name)

    for utt in tqdm(metadata):
        uid = utt["Uid"]
        text = utt["Text"]

        phone_seq = phone_extractor.extract_phone(text)

        phone_path = os.path.join(out_path, uid + ".phone")
        with open(phone_path, "w") as fin:
            fin.write(" ".join(phone_seq))

    if cfg.preprocess.phone_extractor != "lexicon":
        phone_extractor.save_dataset_phone_symbols_to_table()


def save_all_dataset_phone_symbols_to_table(self, cfg, dataset):
    #  phone symbols dict
    phone_symbols = set()

    for dataset_name in dataset:
        phone_symbols_file = os.path.join(
            cfg.preprocess.processed_dir, dataset_name, cfg.preprocess.symbols_dict
        )

        # load and merge saved phone symbols
        assert os.path.exists(phone_symbols_file)
        phone_symbol_dict_saved = SymbolTable.from_file(
            phone_symbols_file
        )._sym2id.keys()
        phone_symbols.update(set(phone_symbol_dict_saved))

    # save all phone symbols to each dataset
    phone_symbol_dict = SymbolTable()
    for s in sorted(list(phone_symbols)):
        phone_symbol_dict.add(s)
    for dataset_name in dataset:
        phone_symbols_file = os.path.join(
            cfg.preprocess.processed_dir, dataset_name, cfg.preprocess.symbols_dict
        )
        phone_symbol_dict.to_file(phone_symbols_file)
