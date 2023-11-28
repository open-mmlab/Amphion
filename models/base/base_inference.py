# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vocoders.vocoder_inference import synthesis
from torch.utils.data import DataLoader
from utils.util import set_all_random_seed
from utils.util import load_config


def parse_vocoder(vocoder_dir):
    r"""Parse vocoder config"""
    vocoder_dir = os.path.abspath(vocoder_dir)
    ckpt_list = [ckpt for ckpt in Path(vocoder_dir).glob("*.pt")]
    ckpt_list.sort(key=lambda x: int(x.stem), reverse=True)
    ckpt_path = str(ckpt_list[0])
    vocoder_cfg = load_config(os.path.join(vocoder_dir, "args.json"), lowercase=True)
    vocoder_cfg.model.bigvgan = vocoder_cfg.vocoder
    return vocoder_cfg, ckpt_path


class BaseInference(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.model_type = cfg.model_type
        self.avg_rtf = list()
        set_all_random_seed(10086)
        os.makedirs(args.output_dir, exist_ok=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(10)  # inference on 1 core cpu.

        # Load acoustic model
        self.model = self.create_model().to(self.device)
        state_dict = self.load_state_dict()
        self.load_model(state_dict)
        self.model.eval()

        # Load vocoder model if necessary
        if self.args.checkpoint_dir_vocoder is not None:
            self.get_vocoder_info()

    def create_model(self):
        raise NotImplementedError

    def load_state_dict(self):
        self.checkpoint_file = self.args.checkpoint_file
        if self.checkpoint_file is None:
            assert self.args.checkpoint_dir is not None
            checkpoint_path = os.path.join(self.args.checkpoint_dir, "checkpoint")
            checkpoint_filename = open(checkpoint_path).readlines()[-1].strip()
            self.checkpoint_file = os.path.join(
                self.args.checkpoint_dir, checkpoint_filename
            )

        self.checkpoint_dir = os.path.split(self.checkpoint_file)[0]

        print("Restore acoustic model from {}".format(self.checkpoint_file))
        raw_state_dict = torch.load(self.checkpoint_file, map_location=self.device)
        self.am_restore_step = re.findall(r"step-(.+?)_loss", self.checkpoint_file)[0]

        return raw_state_dict

    def load_model(self, model):
        raise NotImplementedError

    def get_vocoder_info(self):
        self.checkpoint_dir_vocoder = self.args.checkpoint_dir_vocoder
        self.vocoder_cfg = os.path.join(
            os.path.dirname(self.checkpoint_dir_vocoder), "args.json"
        )
        self.cfg.vocoder = load_config(self.vocoder_cfg, lowercase=True)
        self.vocoder_tag = self.checkpoint_dir_vocoder.split("/")[-2].split(":")[-1]
        self.vocoder_steps = self.checkpoint_dir_vocoder.split("/")[-1].split(".")[0]

    def build_test_utt_data(self):
        raise NotImplementedError

    def build_testdata_loader(self, args, target_speaker=None):
        datasets, collate = self.build_test_dataset()
        self.test_dataset = datasets(self.cfg, args, target_speaker)
        self.test_collate = collate(self.cfg)
        self.test_batch_size = min(
            self.cfg.train.batch_size, len(self.test_dataset.metadata)
        )
        test_loader = DataLoader(
            self.test_dataset,
            collate_fn=self.test_collate,
            num_workers=self.args.num_workers,
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        return test_loader

    def inference_each_batch(self, batch_data):
        raise NotImplementedError

    def inference_for_batches(self, args, target_speaker=None):
        ###### Construct test_batch ######
        loader = self.build_testdata_loader(args, target_speaker)

        n_batch = len(loader)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(
            "Model eval time: {}, batch_size = {}, n_batch = {}".format(
                now, self.test_batch_size, n_batch
            )
        )
        self.model.eval()

        ###### Inference for each batch ######
        pred_res = []
        with torch.no_grad():
            for i, batch_data in enumerate(loader if n_batch == 1 else tqdm(loader)):
                # Put the data to device
                for k, v in batch_data.items():
                    batch_data[k] = batch_data[k].to(self.device)

                y_pred, stats = self.inference_each_batch(batch_data)

                pred_res += y_pred

        return pred_res

    def inference(self, feature):
        raise NotImplementedError

    def synthesis_by_vocoder(self, pred):
        audios_pred = synthesis(
            self.vocoder_cfg,
            self.checkpoint_dir_vocoder,
            len(pred),
            pred,
        )
        return audios_pred

    def __call__(self, utt):
        feature = self.build_test_utt_data(utt)
        start_time = time.time()
        with torch.no_grad():
            outputs = self.inference(feature)[0]
        time_used = time.time() - start_time
        rtf = time_used / (
            outputs.shape[1]
            * self.cfg.preprocess.hop_size
            / self.cfg.preprocess.sample_rate
        )
        print("Time used: {:.3f}, RTF: {:.4f}".format(time_used, rtf))
        self.avg_rtf.append(rtf)
        audios = outputs.cpu().squeeze().numpy().reshape(-1, 1)
        return audios


def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", help="json files for configurations."
    )
    parser.add_argument("--use_ddp_inference", default=False)
    parser.add_argument("--n_workers", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Worker number for inference dataloader",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint dir including model file and configuration",
    )
    parser.add_argument(
        "--checkpoint_file", help="checkpoint file", type=str, default=None
    )
    parser.add_argument(
        "--test_list", help="test utterance list for testing", type=str, default=None
    )
    parser.add_argument(
        "--checkpoint_dir_vocoder",
        help="Vocoder's checkpoint dir including model file and configuration",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir for saving generated results",
    )
    return parser


if __name__ == "__main__":
    parser = base_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Build inference
    inference = BaseInference(cfg, args)
    inference()
