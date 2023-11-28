# This module is from [WeNet](https://github.com/wenet-e2e/wenet).

# ## Citations

# ```bibtex
# @inproceedings{yao2021wenet,
#   title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
#   author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
#   booktitle={Proc. Interspeech},
#   year={2021},
#   address={Brno, Czech Republic },
#   organization={IEEE}
# }

# @article{zhang2022wenet,
#   title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
#   author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
#   journal={arXiv preprint arXiv:2203.15455},
#   year={2022}
# }
#

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.common import IGNORE_ID
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.config import override_config

import onnxruntime as rt
import multiprocessing
import numpy as np

try:
    from swig_decoders import (
        map_batch,
        ctc_beam_search_decoder_batch,
        TrieVector,
        PathTrie,
    )
except ImportError:
    print(
        "Please install ctc decoders first by refering to\n"
        + "https://github.com/Slyne/ctc_decoder.git"
    )
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="recognize with your model")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--test_data", required=True, help="test data file")
    parser.add_argument(
        "--data_type",
        default="raw",
        choices=["raw", "shard"],
        help="train and cv data type",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id for this rank, -1 for cpu"
    )
    parser.add_argument("--dict", required=True, help="dict file")
    parser.add_argument("--encoder_onnx", required=True, help="encoder onnx file")
    parser.add_argument("--decoder_onnx", required=True, help="decoder onnx file")
    parser.add_argument("--result_file", required=True, help="asr result file")
    parser.add_argument("--batch_size", type=int, default=32, help="asr result file")
    parser.add_argument(
        "--mode",
        choices=["ctc_greedy_search", "ctc_prefix_beam_search", "attention_rescoring"],
        default="attention_rescoring",
        help="decoding mode",
    )
    parser.add_argument(
        "--bpe_model", default=None, type=str, help="bpe model for english part"
    )
    parser.add_argument(
        "--override_config", action="append", default=[], help="override yaml config"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="whether to export fp16 model, default false",
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    reverse_weight = configs["model_conf"].get("reverse_weight", 0.0)
    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs["dataset_conf"])
    test_conf["filter_conf"]["max_length"] = 102400
    test_conf["filter_conf"]["min_length"] = 0
    test_conf["filter_conf"]["token_max_length"] = 102400
    test_conf["filter_conf"]["token_min_length"] = 0
    test_conf["filter_conf"]["max_output_input_ratio"] = 102400
    test_conf["filter_conf"]["min_output_input_ratio"] = 0
    test_conf["speed_perturb"] = False
    test_conf["spec_aug"] = False
    test_conf["spec_sub"] = False
    test_conf["spec_trim"] = False
    test_conf["shuffle"] = False
    test_conf["sort"] = False
    test_conf["fbank_conf"]["dither"] = 0.0
    test_conf["batch_conf"]["batch_type"] = "static"
    test_conf["batch_conf"]["batch_size"] = args.batch_size

    test_dataset = Dataset(
        args.data_type,
        args.test_data,
        symbol_table,
        test_conf,
        args.bpe_model,
        partition=False,
    )

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        EP_list = ["CPUExecutionProvider"]

    encoder_ort_session = rt.InferenceSession(args.encoder_onnx, providers=EP_list)
    decoder_ort_session = None
    if args.mode == "attention_rescoring":
        decoder_ort_session = rt.InferenceSession(args.decoder_onnx, providers=EP_list)

    # Load dict
    vocabulary = []
    char_dict = {}
    with open(args.dict, "r") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
            vocabulary.append(arr[0])
    eos = sos = len(char_dict) - 1
    with torch.no_grad(), open(args.result_file, "w") as fout:
        for _, batch in enumerate(test_data_loader):
            keys, feats, _, feats_lengths, _ = batch
            feats, feats_lengths = feats.numpy(), feats_lengths.numpy()
            if args.fp16:
                feats = feats.astype(np.float16)
            ort_inputs = {
                encoder_ort_session.get_inputs()[0].name: feats,
                encoder_ort_session.get_inputs()[1].name: feats_lengths,
            }
            ort_outs = encoder_ort_session.run(None, ort_inputs)
            (
                encoder_out,
                encoder_out_lens,
                ctc_log_probs,
                beam_log_probs,
                beam_log_probs_idx,
            ) = ort_outs
            beam_size = beam_log_probs.shape[-1]
            batch_size = beam_log_probs.shape[0]
            num_processes = min(multiprocessing.cpu_count(), batch_size)
            if args.mode == "ctc_greedy_search":
                if beam_size != 1:
                    log_probs_idx = beam_log_probs_idx[:, :, 0]
                batch_sents = []
                for idx, seq in enumerate(log_probs_idx):
                    batch_sents.append(seq[0 : encoder_out_lens[idx]].tolist())
                hyps = map_batch(batch_sents, vocabulary, num_processes, True, 0)
            elif args.mode in ("ctc_prefix_beam_search", "attention_rescoring"):
                batch_log_probs_seq_list = beam_log_probs.tolist()
                batch_log_probs_idx_list = beam_log_probs_idx.tolist()
                batch_len_list = encoder_out_lens.tolist()
                batch_log_probs_seq = []
                batch_log_probs_ids = []
                batch_start = []  # only effective in streaming deployment
                batch_root = TrieVector()
                root_dict = {}
                for i in range(len(batch_len_list)):
                    num_sent = batch_len_list[i]
                    batch_log_probs_seq.append(batch_log_probs_seq_list[i][0:num_sent])
                    batch_log_probs_ids.append(batch_log_probs_idx_list[i][0:num_sent])
                    root_dict[i] = PathTrie()
                    batch_root.append(root_dict[i])
                    batch_start.append(True)
                score_hyps = ctc_beam_search_decoder_batch(
                    batch_log_probs_seq,
                    batch_log_probs_ids,
                    batch_root,
                    batch_start,
                    beam_size,
                    num_processes,
                    0,
                    -2,
                    0.99999,
                )
                if args.mode == "ctc_prefix_beam_search":
                    hyps = []
                    for cand_hyps in score_hyps:
                        hyps.append(cand_hyps[0][1])
                    hyps = map_batch(hyps, vocabulary, num_processes, False, 0)
            if args.mode == "attention_rescoring":
                ctc_score, all_hyps = [], []
                max_len = 0
                for hyps in score_hyps:
                    cur_len = len(hyps)
                    if len(hyps) < beam_size:
                        hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                    cur_ctc_score = []
                    for hyp in hyps:
                        cur_ctc_score.append(hyp[0])
                        all_hyps.append(list(hyp[1]))
                        if len(hyp[1]) > max_len:
                            max_len = len(hyp[1])
                    ctc_score.append(cur_ctc_score)
                if args.fp16:
                    ctc_score = np.array(ctc_score, dtype=np.float16)
                else:
                    ctc_score = np.array(ctc_score, dtype=np.float32)
                hyps_pad_sos_eos = (
                    np.ones((batch_size, beam_size, max_len + 2), dtype=np.int64)
                    * IGNORE_ID
                )
                r_hyps_pad_sos_eos = (
                    np.ones((batch_size, beam_size, max_len + 2), dtype=np.int64)
                    * IGNORE_ID
                )
                hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
                k = 0
                for i in range(batch_size):
                    for j in range(beam_size):
                        cand = all_hyps[k]
                        l = len(cand) + 2
                        hyps_pad_sos_eos[i][j][0:l] = [sos] + cand + [eos]
                        r_hyps_pad_sos_eos[i][j][0:l] = [sos] + cand[::-1] + [eos]
                        hyps_lens_sos[i][j] = len(cand) + 1
                        k += 1
                decoder_ort_inputs = {
                    decoder_ort_session.get_inputs()[0].name: encoder_out,
                    decoder_ort_session.get_inputs()[1].name: encoder_out_lens,
                    decoder_ort_session.get_inputs()[2].name: hyps_pad_sos_eos,
                    decoder_ort_session.get_inputs()[3].name: hyps_lens_sos,
                    decoder_ort_session.get_inputs()[-1].name: ctc_score,
                }
                if reverse_weight > 0:
                    r_hyps_pad_sos_eos_name = decoder_ort_session.get_inputs()[4].name
                    decoder_ort_inputs[r_hyps_pad_sos_eos_name] = r_hyps_pad_sos_eos
                best_index = decoder_ort_session.run(None, decoder_ort_inputs)[0]
                best_sents = []
                k = 0
                for idx in best_index:
                    cur_best_sent = all_hyps[k : k + beam_size][idx]
                    best_sents.append(cur_best_sent)
                    k += beam_size
                hyps = map_batch(best_sents, vocabulary, num_processes)

            for i, key in enumerate(keys):
                content = hyps[i]
                logging.info("{} {}".format(key, content))
                fout.write("{} {}\n".format(key, content))


if __name__ == "__main__":
    main()
