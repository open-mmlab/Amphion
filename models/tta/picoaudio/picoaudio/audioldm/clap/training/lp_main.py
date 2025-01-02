from cmath import cos
from inspect import getargs
import logging
import os
import random
from datetime import datetime
import bisect
import copy
from sched import scheduler
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.cuda.amp import GradScaler
import faulthandler
import pathlib
import argparse
import time

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, create_model
from training.data import get_data
from training.params import parse_args
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.scheduler import cosine_lr
from training.lp_train import train_one_epoch, evaluate
from open_clip.utils import get_tar_path_from_dataset_name, dataset_split, get_optimizer
from open_clip.utils import load_p, load_class_label
from open_clip.linear_probe import LinearProbe


def maintain_ckpts(args, startidx, all_idx_len):
    for i in reversed(range(startidx, all_idx_len)):
        if os.path.exists(os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt")):
            os.rename(
                os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt"),
                os.path.join(args.checkpoint_path, f"epoch_top_{i+1}.pt"),
            )
    if os.path.exists(
        os.path.join(args.checkpoint_path, f"epoch_top_{all_idx_len}.pt")
    ):
        os.remove(os.path.join(args.checkpoint_path, f"epoch_top_{all_idx_len}.pt"))
    return


def update_top_k_performance(
    new_metrics_inputs, current_top_k_ckpt_metrics, args, ckpt, bignumbetter=True
):
    """
    Record the top-k performance of the current epoch.
    current_top_k_metrics is a dictionary of the form: {1: top_1_ckpt_measure, 2: top_2_ckpt_measure, ...}
    """
    if isinstance(new_metrics_inputs, (list, tuple)):
        new_metrics_inputs = np.mean(new_metrics_inputs)
        return update_top_k_performance(
            new_metrics_inputs,
            current_top_k_ckpt_metrics,
            args=args,
            ckpt=ckpt,
            bignumbetter=bignumbetter,
        )
    elif isinstance(new_metrics_inputs, dict):
        new_metrics_inputs = np.mean(list(new_metrics_inputs.values()))
        return update_top_k_performance(
            new_metrics_inputs,
            current_top_k_ckpt_metrics,
            args=args,
            ckpt=ckpt,
            bignumbetter=bignumbetter,
        )
    elif isinstance(new_metrics_inputs, (float, int)):
        update_flag = {k: False for k in current_top_k_ckpt_metrics.keys()}
        sorted_keys = sorted(current_top_k_ckpt_metrics.keys())
        sorted_values = sorted(
            current_top_k_ckpt_metrics.values(), reverse=bignumbetter
        )
        sorted_values_ = copy.deepcopy(sorted_values)
        sorted_values.append(new_metrics_inputs)
        sorted_values = sorted(sorted_values, reverse=bignumbetter)
        sorted_values = sorted_values[:-1]

        if sorted_values == sorted_values_:
            return current_top_k_ckpt_metrics, new_metrics_inputs
        else:
            for i in range(len(sorted_keys)):
                if current_top_k_ckpt_metrics[sorted_keys[i]] != sorted_values[i]:
                    current_top_k_ckpt_metrics[sorted_keys[i]] = sorted_values[i]
                    update_flag[sorted_keys[i]] = True
            for i in range(len(update_flag)):
                if update_flag[i]:
                    maintain_ckpts(args, i, len(sorted_keys))
                    torch.save(
                        ckpt,
                        os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt"),
                    )
                    break
            return current_top_k_ckpt_metrics, new_metrics_inputs


# def updateifNone(a, b):
#     a = b if None else a
#     return a


def is_pretrained_params(n):
    return (
        n.startswith("clap_model.transformer")
        or n in ["clap_model.positional_embedding", "clap_model.text_projection"]
        or n.startswith("clap_model.token_embedding")
        or n.startswith("clap_model.ln_final")
        or n.startswith("clap_model.logit_scale_t")
    )


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def config_lp_optimizer(model, data, args):
    # set wd-related params to 0 if use adam optimizer
    if args.optimizer == "adam":
        args.wd = 0
        args.wd_pretrained = 0
        args.wd_new = 0

    in_clap = lambda n, p: n.startswith("clap_model")

    named_parameters = list(model.named_parameters())

    optimizer = {}
    scheduler = {}

    # freeze text encoder
    text_freeze_parameters = [
        p
        for n, p in named_parameters
        if n.startswith("clap_model.transformer")
        or n in ["clap_model.positional_embedding", "clap_model.text_projection"]
        or n.startswith("clap_model.token_embedding")
        or n.startswith("clap_model.ln_final")
    ]

    if args.freeze_text:
        logging.info("Freeze Text!!!!")
        for k in text_freeze_parameters:
            k.requires_grad = False

    if not args.lp_freeze:
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        # (yusong): we do not split the learning rate anymore
        # p for n, p in named_parameters if in_clap(n,p) and exclude(n, p) and p.requires_grad
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        # rest_params = [p for n, p in named_parameters if in_clap(n,p) and include(n, p) and p.requires_grad]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        if args.train_data is None:
            optimizer = None
            scheduler = None
        else:
            total_steps = data["train"].dataloader.num_batches * args.epochs

            if args.split_opt:
                for x in ["lr", "beta1", "beta2", "eps", "wd"]:
                    for y in ["_new", "_pretrained"]:
                        if getattr(args, x + y) is None:
                            setattr(args, x + y, getattr(args, x))

                gain_or_bias_pretrained_params = [
                    p
                    for n, p in named_parameters
                    if (exclude(n, p) and p.requires_grad) and is_pretrained_params(n)
                ]
                rest_pretrained_params = [
                    p
                    for n, p in named_parameters
                    if (include(n, p) and p.requires_grad) and is_pretrained_params(n)
                ]
                gain_or_bias_new_params = [
                    p
                    for n, p in named_parameters
                    if (exclude(n, p) and p.requires_grad)
                    and (not is_pretrained_params(n))
                ]
                rest_new_params = [
                    p
                    for n, p in named_parameters
                    if (include(n, p) and p.requires_grad)
                    and (not is_pretrained_params(n))
                ]

                pretrained_params_optimizer = get_optimizer(
                    [
                        {"params": gain_or_bias_pretrained_params, "weight_decay": 0.0},
                        {
                            "params": rest_pretrained_params,
                            "weight_decay": args.wd_pretrained,
                        },
                    ],
                    lr=args.lr_pretrained,
                    betas=(args.beta1_pretrained, args.beta2_pretrained),
                    eps=args.eps_pretrained,
                    momentum=args.momentum_pretrained,
                    optimizer_name=args.optimizer,
                )
                pretrained_params_scheduler = cosine_lr(
                    pretrained_params_optimizer,
                    args.lr_pretrained,
                    args.warmup,
                    total_steps,
                )

                new_params_optimizer = get_optimizer(
                    [
                        {"params": gain_or_bias_new_params, "weight_decay": 0.0},
                        {"params": rest_new_params, "weight_decay": args.wd_new},
                    ],
                    lr=args.lr_new,
                    betas=(args.beta1_new, args.beta2_new),
                    eps=args.eps_new,
                    momentum=args.momentum_new,
                    optimizer_name=args.optimizer,
                )
                new_params_scheduler = cosine_lr(
                    new_params_optimizer, args.lr_new, args.warmup, total_steps
                )

                optimizer["text"] = pretrained_params_optimizer
                optimizer["audio"] = new_params_optimizer
                scheduler["text"] = pretrained_params_scheduler
                scheduler["audio"] = new_params_scheduler

                if args.horovod:
                    pretrained_params_optimizer = hvd.DistributedOptimizer(
                        pretrained_params_optimizer,
                        named_parameters=model.named_parameters(),
                    )
                    new_params_optimizer = hvd.DistributedOptimizer(
                        new_params_optimizer, named_parameters=model.named_parameters()
                    )
                    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                    hvd.broadcast_optimizer_state(
                        pretrained_params_optimizer, root_rank=0
                    )
                    hvd.broadcast_optimizer_state(new_params_optimizer, root_rank=0)
            else:

                optimizer["clap"] = get_optimizer(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.0},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    momentum=args.momentum,
                    optimizer_name=args.optimizer,
                )
                scheduler["clap"] = cosine_lr(
                    optimizer["clap"], args.lr, args.warmup, total_steps
                )

                if args.horovod:
                    optimizer["clap"] = hvd.DistributedOptimizer(
                        optimizer["clap"], named_parameters=model.named_parameters()
                    )
                    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                    hvd.broadcast_optimizer_state(optimizer["clap"], root_rank=0)

    # linear probe optimizer
    else:
        lp_params = [
            p for n, p in named_parameters if (not in_clap(n, p)) and p.requires_grad
        ]
        lp_optim = get_optimizer(
            lp_params,
            lr=args.lp_lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            momentum=0.9,
            optimizer_name=args.optimizer,
        )
        optimizer["lp"] = lp_optim

    return optimizer, scheduler, text_freeze_parameters


def main():
    args = parse_args()

    time.sleep(args.sleep)

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.amodel = args.amodel.replace("/", "-")
    # download sizes.json file

    # (yusong): the below two lines are for debug
    # print("setting up faulthandler")
    # faulthandler.register(10)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    args.class_index_dict = load_class_label(args.class_label_path)

    # get the name of the experiments
    if args.name is None:
        args.name = "-".join(
            [
                datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
                f"linear_probe" f"model_{args.amodel}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if args.remotedata and is_master(args):
        for dataset_name in args.datasetnames:
            for split in dataset_split[dataset_name]:
                if not os.path.exists(f"./json_files/{dataset_name}/{split}"):
                    os.makedirs(f"./json_files/{dataset_name}/{split}")
                os.system(
                    f"aws s3 cp s3://s-laion-audio/webdataset_tar/{dataset_name}/{split}/sizes.json ./json_files/{dataset_name}/{split}/sizes.json"
                )

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)

        # avoid log dir in same name:
        postfix = 0
        while os.path.exists(args.log_path):
            postfix += 1
            log_base_path_new = log_base_path + "-" + str(postfix)
            os.makedirs(log_base_path_new, exist_ok=True)
            log_filename = f"out-{args.rank}" if args.log_local else "out.log"
            args.log_path = os.path.join(log_base_path_new, log_filename)
            # print(
            #     "Error. Experiment already exists. Use --name {} to specify a new experiment."
            # )
            # return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    if is_master(args):
        args.tensorboard_path = (
            os.path.join(args.logs, args.name, "tensorboard")
            if args.tensorboard
            else ""
        )
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""
        args.checkpoint_path = ""

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ["amp", "fp16", "fp32"]
    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train."
        )

    if args.horovod:
        logging.info(
            f"Running in horovod mode with multiple processes / nodes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    logging.info(f"openai cache dir: {os.path.expanduser(args.openai_model_cache_dir)}")

    # Create CLAP model
    clap_model, clap_model_cfg = create_model(
        args.amodel,
        args.tmodel,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
        skip_params=False,
        pretrained_audio=args.pretrained_audio,
        pretrained_text=args.pretrained_text,
        enable_fusion=args.enable_fusion,
        fusion_type=args.fusion_type,
    )

    args.lp_out_ch = len(list(args.class_index_dict.keys()))
    # Linear Probe
    logging.info(f"linear probe using mlp: {args.lp_mlp}")
    logging.info(f"linear probe using freeze: {args.lp_freeze}")
    logging.info(f"linear probe act layer: {args.lp_act}")
    logging.info(f"linear probe out ch: {args.lp_out_ch}")
    logging.info(f"linear probe learning rate (if applicable): {args.lp_lr}")
    logging.info(f"linear probe loss func: {args.lp_loss}")
    logging.info(f"linear probe lp_metrics: {args.lp_metrics}")

    model = LinearProbe(
        clap_model,
        mlp=args.lp_mlp,
        freeze=args.lp_freeze,
        in_ch=512,
        out_ch=args.lp_out_ch,
        act=args.lp_act,
    )  # in_ch is fixed (i.e., 512)
    model = model.to(device)

    if args.horovod:
        with torch.no_grad():
            for param in model.parameters():
                param.set_(param.contiguous())

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if is_master(args):
        logging.info("Linear Probe CLAP Model:")
        logging.info(f"{str(clap_model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args["static_graph"] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True, **ddp_args
        )

    data = get_data(args, clap_model_cfg)
    assert len(data), "At least one train or eval dataset must be specified."
    if args.trace:
        assert "train" not in data, "Cannot train with traced model"

    optimizer, scheduler, text_freeze_parameters = config_lp_optimizer(
        model, data, args
    )

    scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if "epoch" in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith(
                    "module"
                ):
                    sd = {k[len("module.") :]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if args.split_opt:
                    if optimizer is not None:
                        for k, o_ in optimizer.items():
                            o_.load_state_dict(checkpoint[k + "_" + "optimizer"])
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                logging.info(
                    f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})"
                )
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(
                    f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})"
                )
            if args.freeze_text:
                print("Freeze Text!!!!")
                for k in text_freeze_parameters:
                    k.requires_grad = False
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting wandb.")
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="clap",
            notes=args.wandb_notes,
            name=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log="all")
        wandb.save(params_file)
        logging.debug("Finished loading wandb.")

    if "train" not in data:
        evaluate(model, data, start_epoch, args, writer)
        return
    elif start_epoch == 0 and "val" in data and not args.no_eval:
        evaluate(model, data, 0, args, writer)
    if args.save_top_performance:
        current_top_k_ckpt_metrics = {
            i: 0 for i in range(args.save_top_performance)
        }  # initialize the top-k metric for ckpts to 0

    for epoch in range(start_epoch, args.epochs):
        # freeze the text param after (include) args.freeze_text_after, this is -1 by default
        if epoch == args.freeze_text_after:
            print("Text pretrained parameters are freezed since this epoch.")
            for k in text_freeze_parameters:
                k.requires_grad = False
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        if (
            any(v in data for v in ("val", "imagenet-val", "imagenet-v2"))
            and not args.no_eval
        ):
            metrics = evaluate(model, data, completed_epoch, args, writer)
            if args.save_top_performance:
                top_k_dataset = args.top_k_checkpoint_select_dataset
                top_k_metric = args.top_k_checkpoint_select_metric
                filtered_metrics = [
                    v
                    for k, v in metrics.items()
                    if top_k_metric in k and top_k_dataset in k
                ]  # check all R@10 metrics (all dataset) and use it to update the ckpt
        # Saving checkpoints.
        if args.save_logs:
            opt_dict = {
                k + "_" + "optimizer": v.state_dict() for k, v in optimizer.items()
            }
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
            }
            checkpoint_dict.update(opt_dict)
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )
            if args.save_top_performance and not args.no_eval:
                update_top_k_performance(
                    filtered_metrics,
                    current_top_k_ckpt_metrics,
                    args,
                    checkpoint_dict,
                    bignumbetter=True,
                )

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb")
    )
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
