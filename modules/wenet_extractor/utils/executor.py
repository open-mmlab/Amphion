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

import logging
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_


class Executor:
    def __init__(self):
        self.step = 0

    def train(
        self, model, optimizer, scheduler, data_loader, device, writer, args, scaler
    ):
        """Train one epoch"""
        model.train()
        clip = args.get("grad_clip", 50.0)
        log_interval = args.get("log_interval", 10)
        rank = args.get("rank", 0)
        epoch = args.get("epoch", 0)
        accum_grad = args.get("accum_grad", 1)
        is_distributed = args.get("is_distributed", True)
        use_amp = args.get("use_amp", False)
        logging.info(
            "using accumulate grad, new batch size is {} times"
            " larger than before".format(accum_grad)
        )
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    # autocast context
                    # The more details about amp can be found in
                    # https://pytorch.org/docs/stable/notes/amp_examples.html
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss_dict = model(feats, feats_lengths, target, target_lengths)
                        loss = loss_dict["loss"] / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                num_seen_utts += num_utts
                if batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        writer.add_scalar("train_loss", loss, self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    log_str = "TRAIN Batch {}/{} loss {:.6f} ".format(
                        epoch, batch_idx, loss.item() * accum_grad
                    )
                    for name, value in loss_dict.items():
                        if name != "loss" and value is not None:
                            log_str += "{} {:.6f} ".format(name, value.item())
                    log_str += "lr {:.8f} rank {}".format(lr, rank)
                    logging.debug(log_str)

    def cv(self, model, data_loader, device, args):
        """Cross validation on"""
        model.eval()
        rank = args.get("rank", 0)
        epoch = args.get("epoch", 0)
        log_interval = args.get("log_interval", 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss_dict = model(feats, feats_lengths, target, target_lengths)
                loss = loss_dict["loss"]
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = "CV Batch {}/{} loss {:.6f} ".format(
                        epoch, batch_idx, loss.item()
                    )
                    for name, value in loss_dict.items():
                        if name != "loss" and value is not None:
                            log_str += "{} {:.6f} ".format(name, value.item())
                    log_str += "history loss {:.6f}".format(total_loss / num_seen_utts)
                    log_str += " rank {}".format(rank)
                    logging.debug(log_str)
        return total_loss, num_seen_utts
