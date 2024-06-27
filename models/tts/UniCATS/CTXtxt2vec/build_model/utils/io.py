import sys
import yaml
import torch
import json


def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def load_json_config(path):
    with open(path) as file:
        config = json.load(file)
    return config


def save_config_to_yaml(config, path):
    assert path.endswith('.yaml')
    with open(path, 'w') as f:
        f.write(yaml.dump(config))
        f.close()

def save_config_to_json(config, path):
    assert path.endswith('.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def save_dict_to_json(d, path, indent=None):
    json.dump(d, open(path, 'w'), indent=indent)


def load_dict_from_json(path):
    return json.load(open(path, 'r'))


def write_args(args, path):
    args_dict = dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('_'))
    with open(path, 'a') as args_file:
        args_file.write('==> torch version: {}\n'.format(torch.__version__))
        args_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
        args_file.write('==> Cmd:\n')
        args_file.write(str(sys.argv))
        args_file.write('\n==> args:\n')
        for k, v in sorted(args_dict.items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))
        args_file.close()
