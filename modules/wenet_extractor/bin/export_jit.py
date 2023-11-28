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
import os

import torch
import yaml

from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import init_model


def get_args():
    parser = argparse.ArgumentParser(description="export your script model")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--output_file", default=None, help="output file")
    parser.add_argument(
        "--output_quant_file", default=None, help="output quantized model file"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # No need gpu for model export
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(configs)
    print(model)

    load_checkpoint(model, args.checkpoint)
    # Export jit torch script model

    if args.output_file:
        script_model = torch.jit.script(model)
        script_model.save(args.output_file)
        print("Export model successfully, see {}".format(args.output_file))

    # Export quantized jit torch script model
    if args.output_quant_file:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print(quantized_model)
        script_quant_model = torch.jit.script(quantized_model)
        script_quant_model.save(args.output_quant_file)
        print(
            "Export quantized model successfully, "
            "see {}".format(args.output_quant_file)
        )


if __name__ == "__main__":
    main()
