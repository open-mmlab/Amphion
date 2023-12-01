# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import torchaudio
import argparse

from utils.tokenizer import G2PModule, tokenize_text
from utils.tokenizer import AudioTokenizer, tokenize_audio
from utils.symbol_table import SymbolTable, TextToken
from models.tts.valle.valle import VALLE
from models.tts.base.tts_inferece import TTSInference
from models.tts.valle.valle_dataset import VALLETestDataset, VALLETestCollator

class VALLEInference(TTSInference):
    def __init__(self, args=None, cfg=None):
        TTSInference.__init__(self, args, cfg)

        self.g2p_module = G2PModule(backend=self.cfg.preprocess.text_extractor)
        text_token_path = os.path.join(
            cfg.preprocess.processed_dir,
            cfg.dataset[0],
            cfg.preprocess.symbols_dict
        )        
        unique_tokens = SymbolTable.from_file(text_token_path)
        self.text_tokenizer = TextToken(unique_tokens.symbols, add_bos=True, add_eos=True)
        self.audio_tokenizer = AudioTokenizer()


    def _build_model(self):
        model = VALLE(self.cfg.model)
        return model

    def _build_test_dataset(self):
        return VALLETestDataset, VALLETestCollator


    def inference_one_clip(self, text, text_prompt, audio_file, save_name="pred"):
        # extract text token
        phone_seq = tokenize_text(self.g2p_module, text=f"{text_prompt} {text}".strip()) 
        text_tokens, text_tokens_lens = self.text_tokenizer.get_token_id_seq(phone_seq) 

        text_tokens = np.array([text_tokens],dtype=np.int64)
        text_tokens = torch.from_numpy(text_tokens).to(self.device)
        text_tokens_lens = torch.IntTensor([text_tokens_lens]).to(self.device)  
        

        # extract acoustic token
        encoded_frames = tokenize_audio(self.audio_tokenizer, audio_file)
        audio_prompt_token = encoded_frames[0][0].transpose(2, 1).to(self.device)

        # copysyn
        if self.args.copysyn:
            samples = self.audio_tokenizer.decode(encoded_frames)
            audio_copysyn = samples[0].cpu().detach()
            
            out_path = os.path.join(self.args.output_dir, self.infer_type, f"{save_name}_copysyn.wav")
            torchaudio.save(out_path, 
                            audio_copysyn, 
                            self.cfg.preprocess.sampling_rate
            )        


        if self.args.continual:
            encoded_frames = self.model.continual(
                text_tokens,
                text_tokens_lens,
                audio_prompt_token,
            )
        else:
            enroll_x_lens = None
            if text_prompt:
                prompt_phone_seq = tokenize_text(self.g2p_module, text=f"{text_prompt}".strip())
                _, enroll_x_lens = self.text_tokenizer.get_token_id_seq(prompt_phone_seq)
                  
            encoded_frames = self.model.inference(
                text_tokens,
                text_tokens_lens,
                audio_prompt_token,
                enroll_x_lens=enroll_x_lens,
                top_k=self.args.top_k,
                temperature=self.args.temperature,
            )

        samples = self.audio_tokenizer.decode(
            [(encoded_frames.transpose(2, 1), None)]
        )
        
        audio = samples[0].squeeze(0).cpu().detach()
          
        return audio        
              
          
    def inference_for_single_utterance(self):
        text = self.args.text
        text_prompt = self.args.text_prompt
        audio_file = self.args.audio_prompt
        
        if not self.args.continual:
            assert text != ""
        else:
            text = ""
        assert text_prompt != ""
        assert audio_file != ""
        
        audio = self.inference_one_clip(text, text_prompt, audio_file)
        
        return audio
        
    
    def inference_for_batches(self):
        test_list_file = self.args.test_list_file
        assert test_list_file is not None
        
        pred_res = []
        with open(test_list_file, "r") as fin:
            for idx, line in enumerate(fin.readlines()):
                fields = line.strip().split("|")
                if self.args.continual:
                    assert len(fields) == 2
                    text_prompt, audio_prompt_path = fields
                    text = ""
                else:
                    assert len(fields) == 3
                    text_prompt, audio_prompt_path, text = fields
                    
                audio = self.inference_one_clip(text, text_prompt, audio_prompt_path, str(idx))
                pred_res.append(audio)
                    
        return pred_res
                                
        '''
        TODO: batch inference 
        ###### Construct test_batch ######
        n_batch = len(self.test_dataloader)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(
            "Model eval time: {}, batch_size = {}, n_batch = {}".format(
                now, self.test_batch_size, n_batch
            )
        )       
        
        ###### Inference for each batch ######
        pred_res = []
        with torch.no_grad():
            for i, batch_data in enumerate(
                self.test_dataloader if n_batch == 1 else tqdm(self.test_dataloader)
            ):
                if self.args.continual:
                    encoded_frames = self.model.continual(
                        batch_data["phone_seq"],
                        batch_data["phone_len"],
                        batch_data["acoustic_token"],
                    )
                else:
                    encoded_frames = self.model.inference(
                        batch_data["phone_seq"],
                        batch_data["phone_len"],
                        batch_data["acoustic_token"],
                        enroll_x_lens=batch_data["pmt_phone_len"],
                        top_k=self.args.top_k,
                        temperature=self.args.temperature
                    )
                    
                samples = self.audio_tokenizer.decode(
                    [(encoded_frames.transpose(2, 1), None)]
                )
                
                            
                for idx in range(samples.size(0)):
                    audio = samples[idx].cpu()
                    pred_res.append(audio)
                    
        return pred_res
        '''

    def add_arguments(parser: argparse.ArgumentParser):
            parser.add_argument(
                "--text_prompt",
                type=str,
                default="",
                help="Text prompt that should be aligned with --audio_prompt.",
            )

            parser.add_argument(
                "--audio_prompt",
                type=str,
                default="",
                help="Audio prompt that should be aligned with --text_prompt.",
            )
            parser.add_argument(
                "--top-k",
                type=int,
                default=-100,
                help="Whether AR Decoder do top_k(if > 0) sampling.",
            )

            parser.add_argument(
                "--temperature",
                type=float,
                default=1.0,
                help="The temperature of AR Decoder top_k sampling.",
            )

            parser.add_argument(
                "--continual",
                action="store_true", 
                help="Inference for continual task.",
            )

            parser.add_argument(
                "--copysyn",
                action="store_true", 
                help="Copysyn: generate audio by decoder of the original audio tokenizer.",
            )
        
