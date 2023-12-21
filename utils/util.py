# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import collections
import glob
import os
import random
import time
import argparse
from collections import OrderedDict

import json5
import numpy as np
import glob
from torch.nn import functional as F


try:
    from ruamel.yaml import YAML as yaml
except:
    from ruamel_yaml import YAML as yaml

import torch

from utils.hparam import HParams
import logging
from logging import handlers


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def find_checkpoint_of_mapper(mapper_ckpt_dir):
    mapper_ckpts = glob.glob(os.path.join(mapper_ckpt_dir, "ckpts/*.pt"))

    # Select the max steps
    mapper_ckpts.sort()
    mapper_weights_file = mapper_ckpts[-1]
    return mapper_weights_file


def pad_f0_to_tensors(f0s, batched=None):
    # Initialize
    tensors = []

    if batched == None:
        # Get the max frame for padding
        size = -1
        for f0 in f0s:
            size = max(size, f0.shape[-1])

        tensor = torch.zeros(len(f0s), size)

        for i, f0 in enumerate(f0s):
            tensor[i, : f0.shape[-1]] = f0[:]

        tensors.append(tensor)
    else:
        start = 0
        while start + batched - 1 < len(f0s):
            end = start + batched - 1

            # Get the max frame for padding
            size = -1
            for i in range(start, end + 1):
                size = max(size, f0s[i].shape[-1])

            tensor = torch.zeros(batched, size)

            for i in range(start, end + 1):
                tensor[i - start, : f0s[i].shape[-1]] = f0s[i][:]

            tensors.append(tensor)

            start = start + batched

        if start != len(f0s):
            end = len(f0s)

            # Get the max frame for padding
            size = -1
            for i in range(start, end):
                size = max(size, f0s[i].shape[-1])

            tensor = torch.zeros(len(f0s) - start, size)

            for i in range(start, end):
                tensor[i - start, : f0s[i].shape[-1]] = f0s[i][:]

            tensors.append(tensor)

    return tensors


def pad_mels_to_tensors(mels, batched=None):
    """
    Args:
        mels: A list of mel-specs
    Returns:
        tensors: A list of tensors containing the batched mel-specs
        mel_frames: A list of tensors containing the frames of the original mel-specs
    """
    # Initialize
    tensors = []
    mel_frames = []

    # Split mel-specs into batches to avoid cuda memory exceed
    if batched == None:
        # Get the max frame for padding
        size = -1
        for mel in mels:
            size = max(size, mel.shape[-1])

        tensor = torch.zeros(len(mels), mels[0].shape[0], size)
        mel_frame = torch.zeros(len(mels), dtype=torch.int32)

        for i, mel in enumerate(mels):
            tensor[i, :, : mel.shape[-1]] = mel[:]
            mel_frame[i] = mel.shape[-1]

        tensors.append(tensor)
        mel_frames.append(mel_frame)
    else:
        start = 0
        while start + batched - 1 < len(mels):
            end = start + batched - 1

            # Get the max frame for padding
            size = -1
            for i in range(start, end + 1):
                size = max(size, mels[i].shape[-1])

            tensor = torch.zeros(batched, mels[0].shape[0], size)
            mel_frame = torch.zeros(batched, dtype=torch.int32)

            for i in range(start, end + 1):
                tensor[i - start, :, : mels[i].shape[-1]] = mels[i][:]
                mel_frame[i - start] = mels[i].shape[-1]

            tensors.append(tensor)
            mel_frames.append(mel_frame)

            start = start + batched

        if start != len(mels):
            end = len(mels)

            # Get the max frame for padding
            size = -1
            for i in range(start, end):
                size = max(size, mels[i].shape[-1])

            tensor = torch.zeros(len(mels) - start, mels[0].shape[0], size)
            mel_frame = torch.zeros(len(mels) - start, dtype=torch.int32)

            for i in range(start, end):
                tensor[i - start, :, : mels[i].shape[-1]] = mels[i][:]
                mel_frame[i - start] = mels[i].shape[-1]

            tensors.append(tensor)
            mel_frames.append(mel_frame)

    return tensors, mel_frames


def load_model_config(args):
    """Load model configurations (in args.json under checkpoint directory)

    Args:
        args (ArgumentParser): arguments to run bins/preprocess.py

    Returns:
        dict: dictionary that stores model configurations
    """
    if args.checkpoint_dir is None:
        assert args.checkpoint_file is not None
        checkpoint_dir = os.path.split(args.checkpoint_file)[0]
    else:
        checkpoint_dir = args.checkpoint_dir
    config_path = os.path.join(checkpoint_dir, "args.json")
    print("config_path: ", config_path)

    config = load_config(config_path)
    return config


def remove_and_create(dir):
    if os.path.exists(dir):
        os.system("rm -r {}".format(dir))
    os.makedirs(dir, exist_ok=True)


def has_existed(path, warning=False):
    if not warning:
        return os.path.exists(path)

    if os.path.exists(path):
        answer = input(
            "The path {} has existed. \nInput 'y' (or hit Enter) to skip it, and input 'n' to re-write it [y/n]\n".format(
                path
            )
        )
        if not answer == "n":
            return True

    return False


def remove_older_ckpt(saved_model_name, checkpoint_dir, max_to_keep=5):
    if os.path.exists(os.path.join(checkpoint_dir, "checkpoint")):
        with open(os.path.join(checkpoint_dir, "checkpoint"), "r") as f:
            ckpts = [x.strip() for x in f.readlines()]
    else:
        ckpts = []
    ckpts.append(saved_model_name)
    for item in ckpts[:-max_to_keep]:
        if os.path.exists(os.path.join(checkpoint_dir, item)):
            os.remove(os.path.join(checkpoint_dir, item))
    with open(os.path.join(checkpoint_dir, "checkpoint"), "w") as f:
        for item in ckpts[-max_to_keep:]:
            f.write("{}\n".format(item))


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def save_checkpoint(
    args,
    generator,
    g_optimizer,
    step,
    discriminator=None,
    d_optimizer=None,
    max_to_keep=5,
):
    saved_model_name = "model.ckpt-{}.pt".format(step)
    checkpoint_path = os.path.join(args.checkpoint_dir, saved_model_name)

    if discriminator and d_optimizer:
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "global_step": step,
            },
            checkpoint_path,
        )
    else:
        torch.save(
            {
                "generator": generator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "global_step": step,
            },
            checkpoint_path,
        )

    print("Saved checkpoint: {}".format(checkpoint_path))

    if os.path.exists(os.path.join(args.checkpoint_dir, "checkpoint")):
        with open(os.path.join(args.checkpoint_dir, "checkpoint"), "r") as f:
            ckpts = [x.strip() for x in f.readlines()]
    else:
        ckpts = []
    ckpts.append(saved_model_name)
    for item in ckpts[:-max_to_keep]:
        if os.path.exists(os.path.join(args.checkpoint_dir, item)):
            os.remove(os.path.join(args.checkpoint_dir, item))
    with open(os.path.join(args.checkpoint_dir, "checkpoint"), "w") as f:
        for item in ckpts[-max_to_keep:]:
            f.write("{}\n".format(item))


def attempt_to_restore(
    generator, g_optimizer, checkpoint_dir, discriminator=None, d_optimizer=None
):
    checkpoint_list = os.path.join(checkpoint_dir, "checkpoint")
    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readlines()[-1].strip()
        checkpoint_path = os.path.join(checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if generator:
            if not list(generator.state_dict().keys())[0].startswith("module."):
                raw_dict = checkpoint["generator"]
                clean_dict = OrderedDict()
                for k, v in raw_dict.items():
                    if k.startswith("module."):
                        clean_dict[k[7:]] = v
                    else:
                        clean_dict[k] = v
                generator.load_state_dict(clean_dict)
            else:
                generator.load_state_dict(checkpoint["generator"])
        if g_optimizer:
            g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        global_step = 100000
        if discriminator and "discriminator" in checkpoint.keys():
            discriminator.load_state_dict(checkpoint["discriminator"])
            global_step = checkpoint["global_step"]
            print("restore discriminator")
        if d_optimizer and "d_optimizer" in checkpoint.keys():
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])
            print("restore d_optimizer...")
    else:
        global_step = 0
    return global_step


class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def apply_moving_average(model, ema):
    for name, param in model.named_parameters():
        if name in ema.shadow:
            ema.update(name, param.data)


def register_model_to_ema(model, ema):
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)


class YParams(HParams):
    def __init__(self, yaml_file):
        if not os.path.exists(yaml_file):
            raise IOError("yaml file: {} is not existed".format(yaml_file))
        super().__init__()
        self.d = collections.OrderedDict()
        with open(yaml_file) as fp:
            for _, v in yaml().load(fp).items():
                for k1, v1 in v.items():
                    try:
                        if self.get(k1):
                            self.set_hparam(k1, v1)
                        else:
                            self.add_hparam(k1, v1)
                        self.d[k1] = v1
                    except Exception:
                        import traceback

                        print(traceback.format_exc())

    # @property
    def get_elements(self):
        return self.d.items()


def override_config(base_config, new_config):
    """Update new configurations in the original dict with the new dict

    Args:
        base_config (dict): original dict to be overridden
        new_config (dict): dict with new configurations

    Returns:
        dict: updated configuration dict
    """
    for k, v in new_config.items():
        if type(v) == dict:
            if k not in base_config.keys():
                base_config[k] = {}
            base_config[k] = override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def get_lowercase_keys_config(cfg):
    """Change all keys in cfg to lower case

    Args:
        cfg (dict): dictionary that stores configurations

    Returns:
        dict: dictionary that stores configurations
    """
    updated_cfg = dict()
    for k, v in cfg.items():
        if type(v) == dict:
            v = get_lowercase_keys_config(v)
        updated_cfg[k.lower()] = v
    return updated_cfg


def _load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary

    Args:
        config_fn (str): path to configuration file
        lowercase (bool, optional): whether changing keys to lower case. Defaults to False.

    Returns:
        dict: dictionary that stores configurations
    """
    with open(config_fn, "r") as f:
        data = f.read()
    config_ = json5.loads(data)
    if "base_config" in config_:
        # load configurations from new path
        p_config_path = os.path.join(os.getenv("WORK_DIR"), config_["base_config"])
        p_config_ = _load_config(p_config_path)
        config_ = override_config(p_config_, config_)
    if lowercase:
        # change keys in config_ to lower case
        config_ = get_lowercase_keys_config(config_)
    return config_


def load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary

    Args:
        config_fn (str): path to configuration file
        lowercase (bool, optional): _description_. Defaults to False.

    Returns:
        JsonHParams: an object that stores configurations
    """
    config_ = _load_config(config_fn, lowercase=lowercase)
    # create an JsonHParams object with configuration dict
    cfg = JsonHParams(**config_)
    return cfg


def save_config(save_path, cfg):
    """Save configurations into a json file

    Args:
        save_path (str): path to save configurations
        cfg (dict): dictionary that stores configurations
    """
    with open(save_path, "w") as f:
        json5.dump(
            cfg, f, ensure_ascii=False, indent=4, quote_keys=True, sort_keys=True
        )


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class ValueWindow:
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1) :] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


class Logger(object):
    def __init__(
        self,
        filename,
        level="info",
        when="D",
        backCount=10,
        fmt="%(asctime)s : %(message)s",
    ):
        self.level_relations = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "crit": logging.CRITICAL,
        }
        if level == "debug":
            fmt = "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding="utf-8"
        )
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)
        self.logger.info(
            "==========================New Starting Here=============================="
        )


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def get_current_time():
    pass


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)
