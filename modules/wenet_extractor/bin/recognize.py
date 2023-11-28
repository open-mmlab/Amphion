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
from wenet.paraformer.search.beam_search import build_beam_search
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model


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
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--dict", required=True, help="dict file")
    parser.add_argument(
        "--non_lang_syms", help="non-linguistic symbol file. One symbol per line."
    )
    parser.add_argument(
        "--beam_size", type=int, default=10, help="beam size for search"
    )
    parser.add_argument("--penalty", type=float, default=0.0, help="length penalty")
    parser.add_argument("--result_file", required=True, help="asr result file")
    parser.add_argument("--batch_size", type=int, default=16, help="asr result file")
    parser.add_argument(
        "--mode",
        choices=[
            "attention",
            "ctc_greedy_search",
            "ctc_prefix_beam_search",
            "attention_rescoring",
            "rnnt_greedy_search",
            "rnnt_beam_search",
            "rnnt_beam_attn_rescoring",
            "ctc_beam_td_attn_rescoring",
            "hlg_onebest",
            "hlg_rescore",
            "paraformer_greedy_search",
            "paraformer_beam_search",
        ],
        default="attention",
        help="decoding mode",
    )

    parser.add_argument(
        "--search_ctc_weight",
        type=float,
        default=1.0,
        help="ctc weight for nbest generation",
    )
    parser.add_argument(
        "--search_transducer_weight",
        type=float,
        default=0.0,
        help="transducer weight for nbest generation",
    )
    parser.add_argument(
        "--ctc_weight",
        type=float,
        default=0.0,
        help="ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode",
    )

    parser.add_argument(
        "--transducer_weight",
        type=float,
        default=0.0,
        help="transducer weight for rescoring weight in "
        "transducer attention rescore mode",
    )
    parser.add_argument(
        "--attn_weight",
        type=float,
        default=0.0,
        help="attention weight for rescoring weight in "
        "transducer attention rescore mode",
    )
    parser.add_argument(
        "--decoding_chunk_size",
        type=int,
        default=-1,
        help="""decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here""",
    )
    parser.add_argument(
        "--num_decoding_left_chunks",
        type=int,
        default=-1,
        help="number of left chunks for decoding",
    )
    parser.add_argument(
        "--simulate_streaming", action="store_true", help="simulate streaming inference"
    )
    parser.add_argument(
        "--reverse_weight",
        type=float,
        default=0.0,
        help="""right to left weight for attention rescoring
                                decode mode""",
    )
    parser.add_argument(
        "--bpe_model", default=None, type=str, help="bpe model for english part"
    )
    parser.add_argument(
        "--override_config", action="append", default=[], help="override yaml config"
    )
    parser.add_argument(
        "--connect_symbol",
        default="",
        type=str,
        help="used to connect the output characters",
    )

    parser.add_argument(
        "--word", default="", type=str, help="word file, only used for hlg decode"
    )
    parser.add_argument(
        "--hlg", default="", type=str, help="hlg file, only used for hlg decode"
    )
    parser.add_argument(
        "--lm_scale",
        type=float,
        default=0.0,
        help="lm scale for hlg attention rescore decode",
    )
    parser.add_argument(
        "--decoder_scale",
        type=float,
        default=0.0,
        help="lm scale for hlg attention rescore decode",
    )
    parser.add_argument(
        "--r_decoder_scale",
        type=float,
        default=0.0,
        help="lm scale for hlg attention rescore decode",
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

    if (
        args.mode
        in [
            "ctc_prefix_beam_search",
            "attention_rescoring",
            "paraformer_beam_search",
        ]
        and args.batch_size > 1
    ):
        logging.fatal(
            "decoding mode {} must be running with batch_size == 1".format(args.mode)
        )
        sys.exit(1)

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

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
    if "fbank_conf" in test_conf:
        test_conf["fbank_conf"]["dither"] = 0.0
    elif "mfcc_conf" in test_conf:
        test_conf["mfcc_conf"]["dither"] = 0.0
    test_conf["batch_conf"]["batch_type"] = "static"
    test_conf["batch_conf"]["batch_size"] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(
        args.data_type,
        args.test_data,
        symbol_table,
        test_conf,
        args.bpe_model,
        non_lang_syms,
        partition=False,
    )

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_model(configs)

    # Load dict
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    model.eval()

    # Build BeamSearchCIF object
    if args.mode == "paraformer_beam_search":
        paraformer_beam_search = build_beam_search(model, args, device)
    else:
        paraformer_beam_search = None

    with torch.no_grad(), open(args.result_file, "w") as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            if args.mode == "attention":
                hyps, _ = model.recognize(
                    feats,
                    feats_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                )
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == "ctc_greedy_search":
                hyps, _ = model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                )
            elif args.mode == "rnnt_greedy_search":
                assert feats.size(0) == 1
                assert "predictor" in configs
                hyps = model.greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                )
            elif args.mode == "rnnt_beam_search":
                assert feats.size(0) == 1
                assert "predictor" in configs
                hyps = model.beam_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    beam_size=args.beam_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    ctc_weight=args.search_ctc_weight,
                    transducer_weight=args.search_transducer_weight,
                )
            elif args.mode == "rnnt_beam_attn_rescoring":
                assert feats.size(0) == 1
                assert "predictor" in configs
                hyps = model.transducer_attention_rescoring(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    beam_size=args.beam_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    ctc_weight=args.ctc_weight,
                    transducer_weight=args.transducer_weight,
                    attn_weight=args.attn_weight,
                    reverse_weight=args.reverse_weight,
                    search_ctc_weight=args.search_ctc_weight,
                    search_transducer_weight=args.search_transducer_weight,
                )
            elif args.mode == "ctc_beam_td_attn_rescoring":
                assert feats.size(0) == 1
                assert "predictor" in configs
                hyps = model.transducer_attention_rescoring(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    beam_size=args.beam_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    ctc_weight=args.ctc_weight,
                    transducer_weight=args.transducer_weight,
                    attn_weight=args.attn_weight,
                    reverse_weight=args.reverse_weight,
                    search_ctc_weight=args.search_ctc_weight,
                    search_transducer_weight=args.search_transducer_weight,
                    beam_search_type="ctc",
                )
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif args.mode == "ctc_prefix_beam_search":
                assert feats.size(0) == 1
                hyp, _ = model.ctc_prefix_beam_search(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                )
                hyps = [hyp]
            elif args.mode == "attention_rescoring":
                assert feats.size(0) == 1
                hyp, _ = model.attention_rescoring(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight,
                )
                hyps = [hyp]
            elif args.mode == "hlg_onebest":
                hyps = model.hlg_onebest(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    hlg=args.hlg,
                    word=args.word,
                    symbol_table=symbol_table,
                )
            elif args.mode == "hlg_rescore":
                hyps = model.hlg_rescore(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    lm_scale=args.lm_scale,
                    decoder_scale=args.decoder_scale,
                    r_decoder_scale=args.r_decoder_scale,
                    hlg=args.hlg,
                    word=args.word,
                    symbol_table=symbol_table,
                )
            elif args.mode == "paraformer_beam_search":
                hyps = model.paraformer_beam_search(
                    feats,
                    feats_lengths,
                    beam_search=paraformer_beam_search,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                )
            elif args.mode == "paraformer_greedy_search":
                hyps = model.paraformer_greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                )
            for i, key in enumerate(keys):
                content = []
                for w in hyps[i]:
                    if w == eos:
                        break
                    content.append(char_dict[w])
                logging.info("{} {}".format(key, args.connect_symbol.join(content)))
                fout.write("{} {}\n".format(key, args.connect_symbol.join(content)))


if __name__ == "__main__":
    main()
