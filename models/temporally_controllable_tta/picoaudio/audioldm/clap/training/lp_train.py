import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import LPLoss, LPMetrics, lp_gather_features
from open_clip.utils import do_mixup, get_mix_lambda
from .distributed import is_master
from .zero_shot import zero_shot_eval


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def train_one_epoch(
    model,
    data,
    epoch,
    optimizer,
    scaler,
    scheduler,
    args,
    tb_writer=None,
    extra_suffix="",
):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    model.train()
    loss = LPLoss(args.lp_loss)

    dataloader, sampler = data["train"].dataloader, data["train"].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # for toy dataset
    if args.dataset_type == "toy":
        dataloader.dataset.generate_queue()

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if isinstance(scheduler, dict):
            for s in scheduler.values():
                s(step)
        else:
            scheduler(step)

        audio = batch  # contains mel_spec, wavform, and longer list
        class_label = batch["class_label"]
        # audio = audio.to(device=device, non_blocking=True)
        class_label = class_label.to(device=device, non_blocking=True)

        if args.mixup:
            # https://github.com/RetroCirce/HTS-Audio-Transformer/blob/main/utils.py#L146
            mix_lambda = torch.from_numpy(
                get_mix_lambda(0.5, len(audio["waveform"]))
            ).to(device)
            class_label = do_mixup(class_label, mix_lambda)
        else:
            mix_lambda = None

        data_time_m.update(time.time() - end)
        if isinstance(optimizer, dict):
            for o_ in optimizer.values():
                o_.zero_grad()
        else:
            optimizer.zero_grad()

        with autocast():
            pred = model(audio, mix_lambda=mix_lambda, device=device)
            total_loss = loss(pred, class_label)

        if isinstance(optimizer, dict):
            if scaler is not None:
                scaler.scale(total_loss).backward()
                for o_ in optimizer.values():
                    if args.horovod:
                        o_.synchronize()
                        scaler.unscale_(o_)
                        with o_.skip_synchronize():
                            scaler.step(o_)
                    else:
                        scaler.step(o_)
                scaler.update()
            else:
                total_loss.backward()
                for o_ in optimizer.values():
                    o_.step()
        else:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).clap_model.logit_scale_a.clamp_(0, math.log(100))
            unwrap_model(model).clap_model.logit_scale_t.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            if isinstance(audio, dict):
                batch_size = len(audio["waveform"])
            else:
                batch_size = len(audio)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            if isinstance(optimizer, dict):
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f} "
                    f"LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]}"
                )
                log_data = {
                    "loss": loss_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "lr": [o_.param_groups[0]["lr"] for o_ in optimizer.values()],
                }
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f} "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "loss": loss_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            for name, val in log_data.items():
                name = f"train{extra_suffix}/{name}"
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    wandb.log({name: val, "step": step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, extra_suffix=""):
    metrics = {}
    if not args.parallel_eval:
        if not is_master(args):
            return metrics
    device = torch.device(args.device)
    model.eval()

    # CHANGE
    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    # metrics.update(zero_shot_metrics)
    if is_master(args):
        print("Evaluating...")
        metric_names = args.lp_metrics.split(",")
        eval_tool = LPMetrics(metric_names=metric_names)

    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    if "val" in data and (
        args.val_frequency
        and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        if args.parallel_eval:
            dataloader, sampler = data["val"].dataloader, data["val"].sampler
            if args.distributed and sampler is not None:
                sampler.set_epoch(epoch)
            samples_per_val = dataloader.num_samples
        else:
            dataloader = data["val"].dataloader
            num_samples = 0
            samples_per_val = dataloader.num_samples

        eval_info = {"pred": [], "target": []}
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                audio = batch  # contains mel_spec, wavform, and longer list
                class_label = batch["class_label"]

                # audio = audio.to(device=device, non_blocking=True)
                class_label = class_label.to(device=device, non_blocking=True)

                with autocast():
                    pred = model(audio, device=device)
                    if args.parallel_eval:
                        pred, class_label = lp_gather_features(
                            pred, class_label, args.world_size, args.horovod
                        )
                    eval_info["pred"].append(pred)
                    eval_info["target"].append(class_label)

                num_samples += class_label.shape[0]

                if (i % 100) == 0:  # and i != 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]"
                    )

            if is_master(args):
                eval_info["pred"] = torch.cat(eval_info["pred"], 0).cpu()
                eval_info["target"] = torch.cat(eval_info["target"], 0).cpu()
                metric_dict = eval_tool.evaluate_mertics(
                    eval_info["pred"], eval_info["target"]
                )
                metrics.update(metric_dict)
                if "epoch" not in metrics.keys():
                    metrics.update({"epoch": epoch})

    if is_master(args):
        if not metrics:
            return metrics

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\n".join(
                ["\t".join([f"{m}: {round(metrics[m], 4):.4f}"]) for m in metrics]
            )
        )
        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val{extra_suffix}/{name}", val, epoch)

            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        if args.wandb:
            assert wandb is not None, "Please install wandb."
            for name, val in metrics.items():
                wandb.log({f"val{extra_suffix}/{name}": val, "epoch": epoch})

        return metrics
    else:
        return metrics
