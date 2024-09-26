#!/usr/bin/env python3
import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy as np
from tqdm import tqdm
from espnet_utils.cli_readers import file_reader_helper
from espnet_utils.cli_utils import get_commandline_args
from espnet_utils.cli_utils import is_scipy_wav_style
from espnet_utils.cli_writers import file_writer_helper


def get_parser():
    parser = argparse.ArgumentParser(
        description="""
        Paste feature files (assuming they have about the same durations,
        see --length-tolerance), appending the features on each frame;
        think of the unix command 'paste'.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')

    parser.add_argument("--length-tolerance", default=0, type=int,
                        help="If length is different, trim as shortest up to a frame  difference of length-tolerance, "
                             "otherwise exclude segment. (int, default = 0)")
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or '
                             'gzip-level(if hdf5)')

    parser.add_argument('rspecifier', nargs="*", type=str,
                        help='List of read specifiers id. e.g. scp:some.scp'
                             'Currently we only support scp-format.')
    parser.add_argument('wspecifier', type=str,
                        help='Write specifier id. e.g. ark:some.ark')
    return parser


def main():
    args = get_parser().parse_args()
    print(args)
    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    assert len(args.rspecifier) >= 2, "Needs at least two read-specifiers to paste feats"
    assert all([specifier.startswith("scp:") for specifier in args.rspecifier]), \
        ("Currently we only support passing rspecifier in scp format."
         "This is because using kaldiio.load_scp, we can ensure the lazy-loading strategy instead of storing all the feats in memory"
         "Although this may sacrifice some speed but in this way arbitrarily large feats can be supported")
    all_state_dicts = [kaldiio.load_scp(specifier.lstrip("scp:")) for specifier in args.rspecifier]
    all_keys = [set(state_dict.keys()) for state_dict in all_state_dicts]
    # for keys in all_keys[1:]:
    #     assert keys == all_keys[0], "Inputs have different keys"
    intersect_keys = set.intersection(*all_keys)

    with file_writer_helper(args.wspecifier,
                            compress=args.compress,
                            compression_method=args.compression_method) as writer:
        for key in tqdm(intersect_keys, desc="Pasting (concatenating) feats"):
            data_list = [specifier[key] for specifier in all_state_dicts]
            current_data = data_list[0]
            for target_data in data_list[1:]:
                current_length = len(current_data)
                target_length = len(target_data)
                if current_length != target_length:
                    if abs(current_length - target_length) > args.length_tolerance:
                        raise RuntimeError(f"The features of {key} have different lengths "
                                           f"such that they cannot be tolerated by {args.length_tolerance}")
                    else:
                        new_length = min(current_length, target_length)
                        current_data = current_data[:new_length]
                        target_data = target_data[:new_length]
                current_data = np.concatenate([current_data, target_data], axis=1)
            writer[key] = current_data


if __name__ == "__main__":
    main()
