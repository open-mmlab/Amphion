import os
import math
import json
import numpy as np
import pandas as pd
import random
import logging
import argparse
import diffusers
import transformers
from transformers import SchedulerType, get_scheduler
from tqdm.auto import tqdm
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import sys
import utils.torch_tools as torch_tools
import models.controllable_diffusion as ConDiffusion
import models.controllable_dataset as ConDataset
from data.filter_data import get_event_list

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a diffusion model for text to audio generation task."
    )
    parser.add_argument(
        "--train_file", "-f", type=str, default="data/meta_data/train.json"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_epochs",
        "-e",
        type=int,
        default=40,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--model_class",
        "-m",
        type=str,
        default="ClapText_Onset_2_Audio_Diffusion",  # TextOnset2AudioDiffusion
        help="name of model_class",
    )
    parser.add_argument(
        "--dataset_class",
        "-dc",
        type=str,
        default="Clap_Onset_2_Audio_Dataset",  # Text_Onset2AudioDataset
        help="name of model_class",
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=10, help="Audio duration."
    )
    parser.add_argument(
        "--num_examples",
        "-n",
        type=int,
        default=-1,
        help="How many examples to use for training.",
    )
    parser.add_argument(
        "--scheduler_name",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Scheduler identifier.",
    )
    parser.add_argument(
        "--unet_model_config",
        type=str,
        default="utils/configs/frequency.json",
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--onset_column",
        type=str,
        default="onset",
        help="The name of the column in the datasets containing the osnet.",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    if True:
        parser.add_argument(
            "--augment",
            action="store_true",
            default=False,
            help="Augment training data.",
        )
        parser.add_argument(
            "--uncondition",
            action="store_true",
            default=False,
            help="10% uncondition for training.",
        )
        parser.add_argument(
            "--weight_decay", type=float, default=1e-8, help="Weight decay to use."
        )
        parser.add_argument(
            "--snr_gamma",
            type=float,
            # default=None,
            default=5.0,
            help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
            "More details here: https://arxiv.org/abs/2303.09556.",
        )
        parser.add_argument(
            "--max_train_steps",
            type=int,
            default=None,
            help="Total number of training steps to perform. If provided, overrides num_epochs.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=4,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--lr_scheduler_type",
            type=SchedulerType,
            default="linear",
            help="The scheduler type to use.",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        )
        parser.add_argument(
            "--num_warmup_steps",
            type=int,
            default=0,
            help="Number of steps for the warmup in the lr scheduler.",
        )
        parser.add_argument(
            "--adam_beta1",
            type=float,
            default=0.9,
            help="The beta1 parameter for the Adam optimizer.",
        )
        parser.add_argument(
            "--adam_beta2",
            type=float,
            default=0.999,
            help="The beta2 parameter for the Adam optimizer.",
        )
        parser.add_argument(
            "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
        )
        parser.add_argument(
            "--adam_epsilon",
            type=float,
            default=1e-08,
            help="Epsilon value for the Adam optimizer",
        )
        parser.add_argument(
            "--seed", type=int, default=0, help="A seed for reproducible training."
        )
        parser.add_argument(
            "--checkpointing_steps",
            type=str,
            default="best",
            help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
        )
        parser.add_argument(
            "--save_every",
            type=int,
            default=40,
            help="Save model after every how many epochs when checkpointing_steps is set to best.",
        )
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="If the training should continue from a local checkpoint folder.",
        )
        parser.add_argument(
            "--with_tracking",
            action="store_true",
            help="Whether to enable experiment trackers for logging.",
        )
        parser.add_argument(
            "--report_to",
            type=str,
            default="all",
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
                "Only applicable when `--with_tracking` is passed."
            ),
        )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    args.event_list = get_event_list()
    print(args)
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    set_seed(args.seed)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = f"ckpts/{args.model_class}_{args.dataset_class}/base"
        elif args.output_dir is not None:
            args.output_dir = (
                f"ckpts/{args.model_class}_{args.dataset_class}/" + args.output_dir
            )
        os.makedirs(args.output_dir, exist_ok=True)

        with open("{}/summary.jsonl".format(args.output_dir), "w") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

    accelerator.wait_for_everyone()

    # Initialize models
    pretrained_model_name = "audioldm-s-full"
    vae, stft = ConDiffusion.build_pretrained_models(pretrained_model_name)
    # vae, stft, clap, _ =  build_vae_stft_clap_models(pretrained_model_name)

    model = getattr(ConDiffusion, args.model_class)(
        scheduler_name=args.scheduler_name,
        unet_model_config_path=args.unet_model_config,
        snr_gamma=args.snr_gamma,
        uncondition=args.uncondition,
    )

    # Get the datasets
    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files={"train": args.train_file})
    with accelerator.main_process_first():
        train_dataset = getattr(ConDataset, args.dataset_class)(
            raw_datasets["train"], args
        )
        accelerator.print(
            "Num instances in train: {}".format(train_dataset.get_num_instances())
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
    )

    # Optimizer
    optimizer_parameters = model.parameters()
    if hasattr(model, "text_encoder"):
        for param in model.text_encoder.parameters():
            param.requires_grad = False
            model.text_encoder.eval()
            optimizer_parameters = model.unet.parameters()
            accelerator.print("Optimizing UNet parameters.")

    num_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))

    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, stft, model, optimizer, lr_scheduler
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

    # Train!
    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint)
            # path = os.path.basename(args.resume_from_checkpoint)
            accelerator.print(
                f"Resumed from local checkpoint: {args.resume_from_checkpoint}"
            )
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last

    # Duration of the audio clips in seconds
    duration, best_loss, best_epoch = args.duration, np.inf, 0

    for epoch in range(starting_epoch, args.num_epochs):
        model.train()
        total_loss = 0
        logger.info(f"train epoch {epoch} begin!")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                device = model.device

                _, onset, event_info, audios, _, _ = (
                    batch  # idx, onset, event_info, audios, caption, onset_str
                )
                target_length = int(duration * 102.4)
                with torch.no_grad():
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    mel, _, waveform = torch_tools.wav_to_fbank(
                        audios, target_length, stft
                    )
                    mel = mel.unsqueeze(1).to(device)
                    true_latent = unwrapped_vae.get_first_stage_encoding(
                        unwrapped_vae.encode_first_stage(mel)
                    )

                loss = model(
                    {"latent": true_latent, "onset": onset, "event_info": event_info},
                    validation_mode=False,
                )
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        logger.info(f"train epoch {epoch} finish!")
        model.uncondition = False

        if accelerator.is_main_process:
            result = {}
            result["epoch"] = (epoch,)
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item() / len(train_dataloader), 4)

            if result["train_loss"] < best_loss:
                best_loss = result["train_loss"]
                best_epoch = epoch
                if args.checkpointing_steps == "best":
                    accelerator.save(
                        accelerator.unwrap_model(model).state_dict(),
                        f"{args.output_dir}/best.pt",
                    )
                    # Save all states -> continue training
                    # accelerator.save_state("{}/{}".format(args.output_dir, "best"))

            result["best_eopch"] = best_epoch
            logger.info(result)
            result["time"] = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

            with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

        if args.with_tracking:
            accelerator.log(result, step=completed_steps)


if __name__ == "__main__":
    main()
