from models.vc.FreeVC.model import SynthesizerTrn, MultiPeriodDiscriminator
from models.vc.FreeVC.data import FreeVCDataset, FreeVCCollate, BucketSampler
from models.vc.FreeVC.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models.vc.FreeVC.commons import slice_segments
from models.vc.FreeVC.train_utils import (
    get_logger,
    load_checkpoint,
    latest_checkpoint_path,
    summarize,
    plot_spectrogram_to_numpy,
    save_checkpoint,
)
from models.vc.FreeVC.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
)
from utils.util import clip_grad_value_
from utils.util import load_config


import os
import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


torch.backends.cudnn.benchmark = True
global_step = 0


def main(cfg, args):
    global global_step

    assert torch.cuda.is_available(), "CPU training is not allowed."

    logger = get_logger(args.log_dir)
    logger.info(cfg)
    writer = SummaryWriter(log_dir=args.log_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(args.log_dir, "eval"))

    torch.manual_seed(cfg.train.seed)

    train_dataset = FreeVCDataset(
        os.path.join(cfg.preprocess.split_dir, "train.txt"), cfg
    )
    train_sampler = BucketSampler(
        train_dataset,
        cfg.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        shuffle=True,
    )
    collate_fn = FreeVCCollate(cfg)
    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )

    eval_dataset = FreeVCDataset(os.path.join(cfg.preprocess.split_dir, "val.txt"), cfg)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        batch_size=cfg.train.batch_size,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    net_g = SynthesizerTrn(
        cfg.data.filter_length // 2 + 1,
        cfg.train.segment_size // cfg.data.hop_length,
        **cfg.model,
    ).cuda()

    net_d = MultiPeriodDiscriminator(cfg.model.use_spectral_norm).cuda()
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        cfg.train.learning_rate,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        cfg.train.learning_rate,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
    )

    try:
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(args.log_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(args.log_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        epoch_str = 1
        global_step = 0
    print(f"global_step: {global_step}")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=cfg.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=cfg.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=cfg.train.fp16_run)

    for epoch in range(epoch_str, cfg.train.epochs + 1):
        train_and_evaluate(
            epoch,
            cfg,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
            [train_loader, eval_loader],
            logger,
            [writer, writer_eval],
        )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    epoch, cfg, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, items in enumerate(train_loader):
        c, spec, y, spk = items
        if cfg.model.use_spk:
            g = spk.cuda(non_blocking=True)
        else:
            g = None
        spec, y = spec.cuda(non_blocking=True), y.cuda(non_blocking=True)
        c = c.cuda(non_blocking=True)

        torch.cuda.synchronize()

        mel = spec_to_mel_torch(
            spec,
            cfg.data.filter_length,
            cfg.data.n_mel_channels,
            cfg.data.sampling_rate,
            cfg.data.mel_fmin,
            cfg.data.mel_fmax,
        )

        with autocast(enabled=cfg.train.fp16_run):
            y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                c, spec, g=g, mel=mel
            )

            y_mel = slice_segments(
                mel, ids_slice, cfg.train.segment_size // cfg.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                cfg.data.filter_length,
                cfg.data.n_mel_channels,
                cfg.data.sampling_rate,
                cfg.data.hop_length,
                cfg.data.win_length,
                cfg.data.mel_fmin,
                cfg.data.mel_fmax,
            )
            y = slice_segments(
                y, ids_slice * cfg.data.hop_length, cfg.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=cfg.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * cfg.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * cfg.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if global_step % cfg.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
            logger.info(
                "Train Epoch: {} [{:.0f}%]".format(
                    epoch, 100.0 * batch_idx / len(train_loader)
                )
            )
            logger.info([x.item() for x in losses] + [global_step, lr])

            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
            }
            scalar_dict.update(
                {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl}
            )

            scalar_dict.update(
                {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
            )
            scalar_dict.update(
                {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
            )
            scalar_dict.update(
                {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
            )
            image_dict = {
                "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                "slice/mel_gen": plot_spectrogram_to_numpy(
                    y_hat_mel[0].data.cpu().numpy()
                ),
                "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            }
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
            )

        if global_step % cfg.train.eval_interval == 0:
            evaluate(cfg, net_g, eval_loader, writer_eval)
            save_checkpoint(
                net_g,
                optim_g,
                cfg.train.learning_rate,
                epoch,
                os.path.join(args.log_dir, "G_{}.pth".format(global_step)),
            )
            save_checkpoint(
                net_d,
                optim_d,
                cfg.train.learning_rate,
                epoch,
                os.path.join(args.log_dir, "D_{}.pth".format(global_step)),
            )
        global_step += 1

    logger.info("====> Epoch: {}".format(epoch))


def evaluate(cfg, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, spec, y, spk = items
            if cfg.model.use_spk:
                g = spk[:1].cuda(0)
            else:
                g = None
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            break
        mel = spec_to_mel_torch(
            spec,
            cfg.data.filter_length,
            cfg.data.n_mel_channels,
            cfg.data.sampling_rate,
            cfg.data.mel_fmin,
            cfg.data.mel_fmax,
        )
        y_hat = generator.infer(c, g=g, mel=mel)

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            cfg.data.filter_length,
            cfg.data.n_mel_channels,
            cfg.data.sampling_rate,
            cfg.data.hop_length,
            cfg.data.win_length,
            cfg.data.mel_fmin,
            cfg.data.mel_fmax,
        )
    image_dict = {
        "gen/mel": plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
        "gt/mel": plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
    }
    audio_dict = {"gen/audio": y_hat[0], "gt/audio": y[0]}
    summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=cfg.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If specified, to resume from the existing checkpoint.",
    )
    parser.add_argument(
        "--resume_from_ckpt_path",
        type=str,
        default="",
        help="The specific checkpoint path that you want to resume from.",
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        default="",
        help="`resume` for loading all the things (including model weights, optimizer, scheduler, and random states). `finetune` for loading only the model weights",
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    args = parser.parse_args()
    args.log_dir = f"ckpts/vc/FreeVC/{args.exp_name}"
    cfg = load_config(args.config)
    main(cfg, args)
