# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import torch
import math
from torch import nn
from models.tts.UniCATS.CTXtxt2vec.build_model.utils.misc import instantiate_from_config
import time
import numpy as np
import os


class Text2Vec(nn.Module):
    def __init__(
            self,
            diffusion_config
    ):
        super().__init__()
        self.transformer = instantiate_from_config(diffusion_config)

    @property
    def device(self):
        return self.transformer.device

    def generate(self, sample_type, *args):
        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1] == 'q':
                self.transformer.p_sample = self.p_sample_with_truncation(self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top":
            self.transformer.predict_start = self.predict_start_with_truncation(self.transformer.predict_start, sample_type.split(',')[0])
        return self.transformer.sample(*args)

    def set_generate_type(self, sample_type):
        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1] == 'q':
                self.transformer.p_sample = self.p_sample_with_truncation(self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top":
            self.transformer.predict_start = self.predict_start_with_truncation(self.transformer.predict_start, sample_type.split(',')[0])

    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            # content_codec = self.content_codec
            # save_path = self.this_save_path

            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k=truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs

            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))

            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True)
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:, 0:1, :], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:, :-1, :]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float() * out + (1 - temp4.float()) * (-70)
                probs = temp5
                return probs

            return wrapper

        else:
            print("wrong sample type")

    def forward(self, batch, **kwargs):
        return self.transformer(batch['label'][0], batch['feat_len'][0], batch['text'][0], batch['text_len'][0], batch['duration'][0], **kwargs)
