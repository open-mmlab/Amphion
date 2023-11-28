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

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    load_trained_modules,
)
from wenet.utils.executor import Executor
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.scheduler import WarmupLR, NoamHoldAnnealing
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model


def get_args():
    parser = argparse.ArgumentParser(description="training your network")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument(
        "--data_type",
        default="raw",
        choices=["raw", "shard"],
        help="train and cv data type",
    )
    parser.add_argument("--train_data", required=True, help="train data file")
    parser.add_argument("--cv_data", required=True, help="cv data file")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id for this local rank, -1 for cpu"
    )
    parser.add_argument("--model_dir", required=True, help="save model dir")
    parser.add_argument("--checkpoint", help="checkpoint model")
    parser.add_argument(
        "--tensorboard_dir", default="tensorboard", help="tensorboard log dir"
    )
    parser.add_argument(
        "--ddp.rank",
        dest="rank",
        default=0,
        type=int,
        help="global rank for distributed training",
    )
    parser.add_argument(
        "--ddp.world_size",
        dest="world_size",
        default=-1,
        type=int,
        help="""number of total processes/gpus for
                        distributed training""",
    )
    parser.add_argument(
        "--ddp.dist_backend",
        dest="dist_backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="distributed backend",
    )
    parser.add_argument(
        "--ddp.init_method", dest="init_method", default=None, help="ddp init method"
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="num of subprocess workers for reading",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Use pinned memory buffers used for reading",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--fp16_grad_sync",
        action="store_true",
        default=False,
        help="Use fp16 gradient sync for ddp",
    )
    parser.add_argument("--cmvn", default=None, help="global cmvn file")
    parser.add_argument(
        "--symbol_table", required=True, help="model unit symbol table for training"
    )
    parser.add_argument(
        "--non_lang_syms", help="non-linguistic symbol file. One symbol per line."
    )
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument(
        "--bpe_model", default=None, type=str, help="bpe model for english part"
    )
    parser.add_argument(
        "--override_config", action="append", default=[], help="override yaml config"
    )
    parser.add_argument(
        "--enc_init",
        default=None,
        type=str,
        help="Pre-trained model to initialize encoder",
    )
    parser.add_argument(
        "--enc_init_mods",
        default="encoder.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules \
                        to initialize ,separated by a comma",
    )
    parser.add_argument("--lfmmi_dir", default="", required=False, help="LF-MMI dir")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Set random seed
    torch.manual_seed(777)
    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    distributed = args.world_size > 1
    if distributed:
        logging.info("training on multiple gpus, this gpu {}".format(args.gpu))
        dist.init_process_group(
            args.dist_backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )

    symbol_table = read_symbol_table(args.symbol_table)

    train_conf = configs["dataset_conf"]
    cv_conf = copy.deepcopy(train_conf)
    cv_conf["speed_perturb"] = False
    cv_conf["spec_aug"] = False
    cv_conf["spec_sub"] = False
    cv_conf["spec_trim"] = False
    cv_conf["shuffle"] = False
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    train_dataset = Dataset(
        args.data_type,
        args.train_data,
        symbol_table,
        train_conf,
        args.bpe_model,
        non_lang_syms,
        True,
    )
    cv_dataset = Dataset(
        args.data_type,
        args.cv_data,
        symbol_table,
        cv_conf,
        args.bpe_model,
        non_lang_syms,
        partition=False,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=None,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
    )
    cv_data_loader = DataLoader(
        cv_dataset,
        batch_size=None,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
    )

    if "fbank_conf" in configs["dataset_conf"]:
        input_dim = configs["dataset_conf"]["fbank_conf"]["num_mel_bins"]
    else:
        input_dim = configs["dataset_conf"]["mfcc_conf"]["num_mel_bins"]
    vocab_size = len(symbol_table)

    # Save configs to model_dir/train.yaml for inference and export
    configs["input_dim"] = input_dim
    configs["output_dim"] = vocab_size
    configs["cmvn_file"] = args.cmvn
    configs["is_json_cmvn"] = True
    configs["lfmmi_dir"] = args.lfmmi_dir

    if args.rank == 0:
        saved_config_path = os.path.join(args.model_dir, "train.yaml")
        with open(saved_config_path, "w") as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init asr model from configs
    model = init_model(configs)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("the number of model params: {:,d}".format(num_params))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    if args.rank == 0:
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(args.model_dir, "init.zip"))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif args.enc_init is not None:
        logging.info("load pretrained encoders: {}".format(args.enc_init))
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    start_epoch = infos.get("epoch", -1) + 1
    cv_loss = infos.get("cv_loss", 0.0)
    step = infos.get("step", -1)

    num_epochs = configs.get("max_epoch", 100)
    model_dir = args.model_dir
    writer = None
    if args.rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:
        assert torch.cuda.is_available()
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )
        device = torch.device("cuda")
        if args.fp16_grad_sync:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default as comm_hooks,
            )

            model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = model.to(device)

    if configs["optim"] == "adam":
        optimizer = optim.Adam(model.parameters(), **configs["optim_conf"])
    elif configs["optim"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), **configs["optim_conf"])
    else:
        raise ValueError("unknown optimizer: " + configs["optim"])
    if configs["scheduler"] == "warmuplr":
        scheduler = WarmupLR(optimizer, **configs["scheduler_conf"])
    elif configs["scheduler"] == "NoamHoldAnnealing":
        scheduler = NoamHoldAnnealing(optimizer, **configs["scheduler_conf"])
    else:
        raise ValueError("unknown scheduler: " + configs["scheduler"])

    final_epoch = None
    configs["rank"] = args.rank
    configs["is_distributed"] = distributed
    configs["use_amp"] = args.use_amp
    if start_epoch == 0 and args.rank == 0:
        save_model_path = os.path.join(model_dir, "init.pt")
        save_checkpoint(model, save_model_path)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        configs["epoch"] = epoch
        lr = optimizer.param_groups[0]["lr"]
        logging.info("Epoch {} TRAIN info lr {}".format(epoch, lr))
        executor.train(
            model,
            optimizer,
            scheduler,
            train_data_loader,
            device,
            writer,
            configs,
            scaler,
        )
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device, configs)
        cv_loss = total_loss / num_seen_utts

        logging.info("Epoch {} CV info cv_loss {}".format(epoch, cv_loss))
        if args.rank == 0:
            save_model_path = os.path.join(model_dir, "{}.pt".format(epoch))
            save_checkpoint(
                model,
                save_model_path,
                {"epoch": epoch, "lr": lr, "cv_loss": cv_loss, "step": executor.step},
            )
            writer.add_scalar("epoch/cv_loss", cv_loss, epoch)
            writer.add_scalar("epoch/lr", lr, epoch)
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, "final.pt")
        os.remove(final_model_path) if os.path.exists(final_model_path) else None
        os.symlink("{}.pt".format(final_epoch), final_model_path)
        writer.close()


if __name__ == "__main__":
    main()
