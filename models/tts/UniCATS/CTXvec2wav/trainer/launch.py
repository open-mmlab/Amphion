#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Distributed process launcher.

This code is modified from https://github.com/pytorch/pytorch/blob/v1.3.0/torch/distributed/launch.py.

"""
import os
import subprocess
import sys

from argparse import ArgumentParser
from argparse import REMAINDER


def parse_args():
    """Parse arguments."""
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utilty that will spawn up "
        "multiple distributed processes"
    )

    # Optional arguments for the launch helper
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="The number of nodes to use for distributed " "training",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The rank of the node for multi-node distributed " "training",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="The number of processes to launch on each node, "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        help="Master node (rank 0)'s address, should be either "
        "the IP address or the hostname of node 0, for "
        "single node multi-proc training, the "
        "--master_addr can simply be 127.0.0.1",
    )
    parser.add_argument(
        "--master_port",
        default=29500,
        type=int,
        help="Master node (rank 0)'s free port that needs to "
        "be used for communciation during distributed "
        "training",
    )
    parser.add_argument(
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "-c",
        "--command",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script " "as a command.",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script/command to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


def main():
    """Launch distributed processes."""
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if "OMP_NUM_THREADS" not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print(
            "*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process "
            "to be {} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************".format(
                current_env["OMP_NUM_THREADS"]
            )
        )

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        if args.command:
            cmd = [args.training_script]
        else:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
            cmd.append(args.training_script)

        if not args.use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(args.training_script_args)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    main()
