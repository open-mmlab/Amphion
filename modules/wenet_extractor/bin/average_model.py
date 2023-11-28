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

import os
import argparse
import glob

import yaml
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description="average model")
    parser.add_argument("--dst_model", required=True, help="averaged model")
    parser.add_argument("--src_path", required=True, help="src model path for average")
    parser.add_argument("--val_best", action="store_true", help="averaged model")
    parser.add_argument("--num", default=5, type=int, help="nums for averaged model")
    parser.add_argument(
        "--min_epoch", default=0, type=int, help="min epoch used for averaging model"
    )
    parser.add_argument(
        "--max_epoch",
        default=65536,
        type=int,
        help="max epoch used for averaging model",
    )

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    checkpoints = []
    val_scores = []
    if args.val_best:
        yamls = glob.glob("{}/[!train]*.yaml".format(args.src_path))
        for y in yamls:
            with open(y, "r") as f:
                dic_yaml = yaml.load(f, Loader=yaml.FullLoader)
                loss = dic_yaml["cv_loss"]
                epoch = dic_yaml["epoch"]
                if epoch >= args.min_epoch and epoch <= args.max_epoch:
                    val_scores += [[epoch, loss]]
        val_scores = np.array(val_scores)
        sort_idx = np.argsort(val_scores[:, -1])
        sorted_val_scores = val_scores[sort_idx][::1]
        print("best val scores = " + str(sorted_val_scores[: args.num, 1]))
        print(
            "selected epochs = "
            + str(sorted_val_scores[: args.num, 0].astype(np.int64))
        )
        path_list = [
            args.src_path + "/{}.pt".format(int(epoch))
            for epoch in sorted_val_scores[: args.num, 0]
        ]
    else:
        path_list = glob.glob("{}/[0-9]*.pt".format(args.src_path))
        path_list = sorted(path_list, key=os.path.getmtime)
        path_list = path_list[-args.num :]
    print(path_list)
    avg = None
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print("Processing {}".format(path))
        states = torch.load(path, map_location=torch.device("cpu"))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print("Saving to {}".format(args.dst_model))
    torch.save(avg, args.dst_model)


if __name__ == "__main__":
    main()
