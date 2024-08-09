import argparse
import os
import warnings
import time
import torch
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.build import build_model
from models.tts.UniCATS.CTXtxt2vec.dataset.build import build_dataloader
from models.tts.UniCATS.CTXtxt2vec.build_model.utils.misc import seed_everything, merge_opts_to_config, modify_config_for_debug
from models.tts.UniCATS.CTXtxt2vec.build_model.utils.io import load_json_config
from models.tts.UniCATS.CTXtxt2vec.build_model.engine.logger import Logger
from models.tts.UniCATS.CTXtxt2vec.build_model.engine.solver import Solver
from models.tts.UniCATS.CTXtxt2vec.trainer.launch import launch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

# environment variables
NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training script')
    parser.add_argument('--config_file', type=str, default='../../../../config/UniCATS_txt2vec.json',
                        help='path of config file')
    parser.add_argument('--name', type=str, default='',
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file')
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results')
    parser.add_argument('--log_frequency', type=int, default=100,
                        help='print frequency (default: 100)')
    parser.add_argument('--load_path', type=str, default=None,
                        help='path to model that need to be loaded, '
                             'used for loading pretrained model')
    parser.add_argument('--resume_name', type=str, default=None,
                        help='resume one experiment with the given name')
    parser.add_argument('--auto_resume', action='store_true',
                        help='automatically resume the training')

    # args for ddp
    parser.add_argument('--num_node', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=NODE_RANK,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default=DIST_URL,
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                             ' used, and ddp will be disabled')
    parser.add_argument('--sync_bn', action='store_true',
                        help='use sync BN layer')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')
    parser.add_argument('--timestamp', action='store_true',  # default=True,
                        help='use tensorboard for logging')
    # args for random
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', action='store_true',
                        help='set cudnn.deterministic True')

    parser.add_argument('--amp', action='store_true',  # default=True,
                        help='automatic mixture of precesion')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='set as debug mode')
    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    if args.resume_name is not None:
        args.name = args.resume_name
        args.config_file = os.path.join(args.output, args.resume_name, 'configs', 'config.json')
        args.auto_resume = True
    else:
        if args.name == '':
            args.name = os.path.basename(args.config_file).replace('.json', '')
        if args.timestamp:
            assert not args.auto_resume, "for timstamp, auto resume is hard to find the save directory"
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            args.name = time_str + '-' + args.name

    # modify args for debugging
    if args.debug:
        args.name = 'debug'
        if args.gpu is None:
            args.gpu = 0

    args.save_dir = os.path.join(args.output, args.name)
    return args


def main():
    args = get_args()

    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

    launch(main_worker, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1

    # load config
    # config = load_yaml_config(args.config_file)
    config = load_json_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    if args.debug:
        config = modify_config_for_debug(config)

    # get logger
    logger = Logger(args)
    logger.save_config(config)

    # get model 
    model = build_model(config, args)
    # print(model)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # get dataloader
    dataloader_info = build_dataloader(config, args)

    # get solver
    solver = Solver(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    # resume 
    if args.load_path is not None:  # only load the model parameters
        solver.resume(path=args.load_path,
                      # load_model=True,
                      load_optimizer_and_scheduler=False,
                      load_others=False)
    if args.auto_resume:
        solver.resume()

    solver.train()


if __name__ == '__main__':
    main()
