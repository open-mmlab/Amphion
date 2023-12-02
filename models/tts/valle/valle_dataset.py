# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.base.base_dataset import (
    BaseCollator,
    BaseDataset,
    BaseTestDataset,
    BaseTestCollator,
)
from utils.symbol_table import SymbolTable, TextToken
from utils.tokenizer import G2PModule, AudioTokenizer, tokenize_text, tokenize_audio

class VALLEDataset(BaseDataset):
    def __init__(self, cfg, dataset, is_valid=False):

        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """

        assert isinstance(dataset, str)

        self.cfg = cfg
        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)

        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)
        self.metadata = self.get_metadata()

        self.data_root = processed_data_dir

        assert cfg.preprocess.use_acoustic_token == True
        if cfg.preprocess.use_acoustic_token:
            self.utt2acousticToken_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2acousticToken_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.acoustic_token_dir,   # code
                    uid + ".npy",
                )

        assert cfg.preprocess.use_phone == True
        if cfg.preprocess.use_phone:
            self.utt2seq = {}
            
            ### get text tokenizer
            text_token_path = os.path.join(
                cfg.preprocess.processed_dir,
                dataset,
                cfg.preprocess.symbols_dict
            )
            unique_tokens = SymbolTable.from_file(text_token_path)
            text_tokenizer = TextToken(unique_tokens.symbols, add_bos=True, add_eos=True)
            
            # convert phone sequence to id sequence
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                
                phone_path = os.path.join(processed_data_dir, 
                                          cfg.preprocess.phone_dir,
                                          uid+'.phone'
                                          )
                with open(phone_path, 'r') as fin:
                    phones = fin.readlines()
                    assert len(phones) == 1
                    phones = phones[0].strip()
                phones = phones.split(' ')

                phone_id_seq, phn_len = text_tokenizer.get_token_id_seq(phones)
                self.utt2seq[utt] = phone_id_seq


    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        metadata_filter = []
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for utt_info in metadata:
            duration = utt_info['Duration']
            if duration >= self.cfg.preprocess.max_duration or duration <= self.cfg.preprocess.min_duration:
                continue
            metadata_filter.append(utt_info)

        return metadata_filter

    def get_dur(self, idx):
        utt_info = self.metadata[idx]
        return utt_info['Duration']


    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        # acoustic token
        if self.cfg.preprocess.use_acoustic_token:
            acoustic_token = np.load(self.utt2acousticToken_path[utt])
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = acoustic_token.shape[0]
            single_feature["acoustic_token"] = acoustic_token  # [T, 8]

        # phoneme sequence todo
        if self.cfg.preprocess.use_phone:
            single_feature["phone_seq"] = np.array(self.utt2seq[utt])
            single_feature["phone_len"] = len(self.utt2seq[utt])


        return single_feature

class VALLECollator(BaseCollator):
    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)
        
    def __call__(self, batch):
        parsed_batch_features = BaseCollator.__call__(self, batch)
        return parsed_batch_features

class VALLETestDataset(BaseTestDataset):
    def __init__(self,args, cfg):
        self.cfg = cfg
        self.args = args
        
        self.g2p_module = G2PModule(backend=self.cfg.preprocess.text_extractor)
        self.dataset = args.dataset if args.dataset is not None else self.cfg.dataset[0]
        
        ### get text tokenizer
        text_token_path = os.path.join(
            cfg.preprocess.processed_dir,
            self.dataset,
            cfg.preprocess.symbols_dict
        )        
        unique_tokens = SymbolTable.from_file(text_token_path)
        self.text_tokenizer = TextToken(unique_tokens.symbols, 
                                        add_bos=True, 
                                        add_eos=True)
                    
        self.audio_tokenizer = AudioTokenizer()

        # construst metadata
        assert args.test_list_file is not None
        if args.test_list_file is not None:
            self.metadata = []

            print('test_list_file: ', args.test_list_file)
            
            with open(args.test_list_file, "r") as fin:
                for idx, line in enumerate(fin.readlines()):
                    utt_info = {}
                    fields = line.strip().split("|")
                    print('fields: ', fields)
                    if self.args.continual:
                        assert len(fields) == 2
                        text_prompt, audio_prompt_path = fields
                        text = ""
                    else:
                        assert len(fields) == 3
                        text_prompt, audio_prompt_path, text = fields

                    utt_info["Dataset"] = "null"
                    utt_info["Uid"] = str(idx)
                    utt_info["Text"] = text
                    utt_info["Text_pormpt"] = text_prompt
                    utt_info["Audio_pormpt_path"] = audio_prompt_path
                                        
                    if cfg.preprocess.use_phone:
                        # convert text to phone sequence
                        phone_seq = tokenize_text(self.g2p_module, text=f"{text_prompt} {text}".strip()) 
                        prompt_phone_seq = tokenize_text(self.g2p_module, text=f"{text_prompt}".strip()) 
                        utt_info["Phone"] = phone_seq
                        utt_info["Prompt_phone"] = prompt_phone_seq
                        
                        
                    self.metadata.append(utt_info)
        # else:
        #     assert args.testing_set
        #     self.metafile_path = os.path.join(
        #         cfg.preprocess.processed_dir,
        #         args.dataset,
        #         "{}.json".format(args.testing_set),
        #     )
        #     self.metadata = self.get_metadata()
            

        # prepare data
        assert cfg.preprocess.use_acoustic_token == True
        if cfg.preprocess.use_acoustic_token:
            self.utt2acousticToken = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                
                # extract acoustic token
                audio_file = utt_info["Audio_pormpt_path"] 
                encoded_frames = tokenize_audio(self.audio_tokenizer, audio_file)
                audio_prompt_token = encoded_frames[0][0].transpose(2, 1).squeeze(0).cpu().numpy()
                self.utt2acousticToken[utt] = audio_prompt_token
                
        
        assert cfg.preprocess.use_phone == True
        if cfg.preprocess.use_phone:
            self.utt2seq = {}
            self.utt2pmtseq = {}
            
            ### get text tokenizer
            text_token_path = os.path.join(
                cfg.preprocess.processed_dir,
                self.dataset,
                cfg.preprocess.symbols_dict
            )
            unique_tokens = SymbolTable.from_file(text_token_path)
            text_tokenizer = TextToken(unique_tokens.symbols, add_bos=True, add_eos=True)
            
            # convert phone sequence to id sequence
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                
                phone_id_seq, phn_len = text_tokenizer.get_token_id_seq(utt_info["Phone"])
                prompt_phone_id_seq, prompt_phn_len = text_tokenizer.get_token_id_seq(utt_info["Prompt_phone"])
                self.utt2seq[utt] = phone_id_seq
                self.utt2pmtseq[utt] = prompt_phone_id_seq



    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()
        
        # acoustic token
        if self.cfg.preprocess.use_acoustic_token:
            acoustic_token = self.utt2acousticToken[utt]
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = acoustic_token.shape[0]
            single_feature["acoustic_token"] = acoustic_token  # [T, 8]

        # phoneme sequence todo
        if self.cfg.preprocess.use_phone:
            single_feature["phone_seq"] = np.array(self.utt2seq[utt])
            single_feature["phone_len"] = len(self.utt2seq[utt])
            single_feature["pmt_phone_seq"] = np.array(self.utt2pmtseq[utt])
            single_feature["pmt_phone_len"] = len(self.utt2pmtseq[utt])

        return single_feature

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata

    def __len__(self):
        return len(self.metadata)

class VALLETestCollator(BaseTestCollator):

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "phone_len":
                packed_batch_features["phone_len"] = torch.LongTensor(
                    [b["phone_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["phone_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["phn_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "pmt_phone_len":
                packed_batch_features["pmt_phone_len"] = torch.LongTensor(
                    [b["pmt_phone_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["pmt_phone_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["pmt_phone_len_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "audio_len":
                packed_batch_features["audio_len"] = torch.LongTensor(
                    [b["audio_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["audio_len"], 1), dtype=torch.long) for b in batch
                ]
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
