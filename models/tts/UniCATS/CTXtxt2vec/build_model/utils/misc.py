import importlib
import random
import numpy as np
import torch
import warnings
import os


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python's random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def merge_opts_to_config(config, opts):
    def modify_dict(c, nl, v):
        if len(nl) == 1:
            c[nl[0]] = type(c[nl[0]])(v)
        else:
            # print(nl)
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)
        return c

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "each opts should be given by the name and values! The length shall be even number!"
        for i in range(len(opts) // 2):
            name = opts[2 * i]
            value = opts[2 * i + 1]
            config = modify_dict(config, name.split('.'), value)
    return config


def modify_config_for_debug(config):
    config['dataloader']['num_workers'] = 0
    config['dataloader']['batch_size'] = 1
    return config


def get_model_parameters_info(model):
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()
            else:
                parameters[child_name]['non_trainable'] += p.numel()
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable']

        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']

    # format the numbers
    def format_number(num):
        K = 2 ** 10
        M = 2 ** 20
        G = 2 ** 30
        if num > G:  # K
            uint = 'G'
            num = round(float(num) / G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num) / M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num) / K, 2)
        else:
            uint = ''

        return '{}{}'.format(num, uint)

    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)

    format_dict(parameters)
    return parameters


def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)

    d = int(h // 24)
    h = h - d * 24

    if d == 0:
        if h == 0:
            if m == 0:
                ft = '{:02d}s'.format(s)
            else:
                ft = '{:02d}m:{:02d}s'.format(m, s)
        else:
            ft = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)

    else:
        ft = '{:d}d:{:02d}h:{:02d}m:{:02d}s'.format(d, h, m, s)

    return ft


def instantiate_from_config(config):
    if config is None:
        return None
    # print(config)
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))


def class_from_string(class_name):
    module, cls = class_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls


def get_all_file(dir, end_with='.h5'):
    if isinstance(end_with, str):
        end_with = [end_with]
    filenames = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            for ew in end_with:
                if f.endswith(ew):
                    filenames.append(os.path.join(root, f))
                    break
    return filenames


def get_sub_dirs(dir, abs=True):
    sub_dirs = os.listdir(dir)
    if abs:
        sub_dirs = [os.path.join(dir, s) for s in sub_dirs]
    return sub_dirs


def get_model_buffer(model):
    state_dict = model.state_dict()
    buffers_ = {}
    params_ = {n: p for n, p in model.named_parameters()}

    for k in state_dict:
        if k not in params_:
            buffers_[k] = state_dict[k]
    return buffers_
