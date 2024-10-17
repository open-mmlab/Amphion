# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing
import os
import subprocess
import time

from utils.logger import Logger
from utils.tool import get_gpu_nums


def run_script(args, gpu_id, self_id):
    """
    Run the script by passing the GPU ID and self ID to environment variables and execute the main.py script.

    Args:
        gpu_id (int): ID of the GPU.
        self_id (int): ID of the process.

    Returns:
        None
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["SELF_ID"] = str(self_id)

    command = (
        f"source {args.conda_path} &&"
        'eval "$(conda shell.bash hook)" && '
        f"conda activate {args.conda_env_name} && "
        "python main.py"
    )

    try:
        process = subprocess.Popen(command, shell=True, env=env, executable="/bin/bash")
        process.wait()
        logger.info(f"Process for GPU {gpu_id} completed successfully.")
    except KeyboardInterrupt:
        logger.warning(f"Multi - GPU {gpu_id}: Interrupted by keyboard, exiting...")
    except Exception as e:
        logger.error(f"Error occurred for GPU {gpu_id}: {e}")


def main(args, self_id):
    """
    Start multiple script tasks using multiple processes, each process using one GPU.

    Args:
        self_id (str): Identifier for the current process.

    Returns:
        None
    """
    disabled_ids = []
    if args.disabled_gpu_ids:
        disabled_ids = [int(i) for i in args.disabled_gpu_ids.split(",")]
        logger.info(f"CUDA_DISABLE_ID is set, not using: {disabled_ids}")

    gpus_count = get_gpu_nums()

    available_gpus = [i for i in range(gpus_count) if i not in disabled_ids]
    processes = []

    for gpu_id in available_gpus:
        process = multiprocessing.Process(
            target=run_script, args=(args, gpu_id, self_id)
        )
        process.start()
        logger.info(f"GPU {gpu_id}: started...")
        time.sleep(1)
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_id", type=str, default="main_multi", help="Log ID")
    parser.add_argument(
        "--disabled_gpu_ids",
        type=str,
        default="",
        help="Comma-separated list of disabled GPU IDs, default uses all available GPUs",
    )
    parser.add_argument(
        "--conda_path",
        type=str,
        default="/opt/conda/etc/profile.d/conda.sh",
        help="Conda path",
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        default="AudioPipeline",
        help="Conda environment name",
    )
    parser.add_argument(
        "--main_command_args",
        type=str,
        default="",
        help="Main command args, check available options by `python main.py --help`",
    )
    args = parser.parse_args()

    self_id = args.self_id
    if "SELF_ID" in os.environ:
        self_id = f"{self_id}_#{os.environ['SELF_ID']}"

    logger = Logger.get_logger(self_id)

    logger.info(f"Starting main_multi.py with self_id: {self_id}, args: {vars(args)}.")
    main(args, self_id)
    logger.info("Exiting main_multi.py...")
