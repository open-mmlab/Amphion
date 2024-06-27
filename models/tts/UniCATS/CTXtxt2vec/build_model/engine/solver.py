# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import os
import time
import math
import torch
import threading
import matplotlib.pyplot as plt
import multiprocessing
import copy
from torch.nn.utils import clip_grad_norm_, clip_grad_norm
from .tool import make_grid
from models.tts.UniCATS.CTXtxt2vec.build_model.utils.misc import instantiate_from_config, format_seconds
from models.tts.UniCATS.CTXtxt2vec.trainer.distributed import reduce_dict
from models.tts.UniCATS.CTXtxt2vec.trainer.distributed import is_primary, get_rank
from models.tts.UniCATS.CTXtxt2vec.build_model.utils.misc import get_model_parameters_info
from models.tts.UniCATS.CTXtxt2vec.build_model.engine.lr_scheduler import ReduceLROnPlateauWithWarmup, CosineAnnealingLRWithWarmup
from models.tts.UniCATS.CTXtxt2vec.build_model.engine.ema import EMA
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP = True
except:
    print('Warning: import torch.amp failed, so no amp will be used!')
    AMP = False

STEP_WITH_LOSS_SCHEDULERS = (ReduceLROnPlateauWithWarmup, ReduceLROnPlateau)


class Solver(object):
    def __init__(self, config, args, model, dataloader, logger):
        self.config = config
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.logger = logger

        self.max_epochs = config['solver']['max_epochs']
        self.save_epochs = config['solver']['save_epochs']
        self.save_iterations = config['solver'].get('save_iterations', -1)
        self.sample_iterations = config['solver']['sample_iterations']
        if self.sample_iterations == 'epoch':
            self.sample_iterations = self.dataloader['train_iterations']
        self.validation_epochs = config['solver'].get('validation_epochs', 2)
        assert isinstance(self.save_epochs, (int, list))
        assert isinstance(self.validation_epochs, (int, list))
        self.debug = config['solver'].get('debug', False)

        self.last_epoch = -1
        self.last_iter = -1
        self.ckpt_dir = os.path.join(args.save_dir, 'checkpoint')
        self.image_dir = os.path.join(args.save_dir, 'images')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        # get grad_clipper
        if 'clip_grad_norm' in config['solver']:
            self.clip_grad_norm = instantiate_from_config(config['solver']['clip_grad_norm'])
        else:
            self.clip_grad_norm = None

        # get lr
        adjust_lr = config['solver'].get('adjust_lr', 'sqrt')
        base_lr = config['solver'].get('base_lr', 1.0e-4)
        if adjust_lr == 'none':
            self.lr = base_lr
        elif adjust_lr == 'sqrt':
            self.lr = base_lr * math.sqrt(args.world_size * config['dataloader']['batch_size'])
        elif adjust_lr == 'linear':
            self.lr = base_lr * args.world_size * config['dataloader']['batch_size']
        else:
            raise NotImplementedError('Unknown type of adjust lr {}!'.format(adjust_lr))
        self.logger.log_info('Get lr {} from base lr {} with {}'.format(self.lr, base_lr, adjust_lr))

        if hasattr(model, 'get_optimizer_and_scheduler') and callable(getattr(model, 'get_optimizer_and_scheduler')):
            optimizer_and_scheduler = model.get_optimizer_and_scheduler(config['solver']['optimizers_and_schedulers'])
        else:
            optimizer_and_scheduler = self._get_optimizer_and_scheduler(config['solver']['optimizers_and_schedulers'])

        assert type(optimizer_and_scheduler) == type({}), 'optimizer and schduler should be a dict!'
        self.optimizer_and_scheduler = optimizer_and_scheduler

        # configre for ema
        if 'ema' in config['solver'] and args.local_rank == 0:
            ema_args = config['solver']['ema']
            ema_args['model'] = self.model
            self.ema = EMA(**ema_args)
        else:
            self.ema = None

        self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.model.cuda()
        self.device = self.model.device
        if self.args.distributed:
            self.logger.log_info('Distributed, begin DDP the model...')
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu])  # , find_unused_parameters=True)
            self.logger.log_info('Distributed, DDP model done!')
        # prepare for amp
        self.args.amp = self.args.amp and AMP
        if self.args.amp:
            self.scaler = GradScaler()
            self.logger.log_info('Using AMP for training!')

        self.logger.log_info("{}: global rank {}: prepare solver done!".format(self.args.name, self.args.global_rank), check_primary=False)

    def _get_optimizer_and_scheduler(self, op_sc_list):
        optimizer_and_scheduler = {}
        for op_sc_cfg in op_sc_list:
            op_sc = {
                'name': op_sc_cfg.get('name', 'none'),
                'start_epoch': op_sc_cfg.get('start_epoch', 0),
                'end_epoch': op_sc_cfg.get('end_epoch', -1),
                'start_iteration': op_sc_cfg.get('start_iteration', 0),
                'end_iteration': op_sc_cfg.get('end_iteration', -1),
            }

            if op_sc['name'] == 'none':
                # parameters = self.model.parameters()
                parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            else:
                # NOTE: get the parameters with the given name, the parameters() should be overide
                parameters = self.model.parameters(name=op_sc['name'])

            # build optimizer
            op_cfg = op_sc_cfg.get('optimizer', {'target': 'torch.optim.SGD', 'params': {}})
            if 'params' not in op_cfg:
                op_cfg['params'] = {}
            if 'lr' not in op_cfg['params']:
                op_cfg['params']['lr'] = self.lr
            op_cfg['params']['params'] = parameters
            optimizer = instantiate_from_config(op_cfg)
            op_sc['optimizer'] = {
                'module': optimizer,
                'step_iteration': op_cfg.get('step_iteration', 1)
            }
            assert isinstance(op_sc['optimizer']['step_iteration'], int), 'optimizer steps should be a integer number of iterations'

            # build scheduler
            if 'scheduler' in op_sc_cfg:
                sc_cfg = op_sc_cfg['scheduler']
                sc_cfg['params']['optimizer'] = optimizer
                # for cosine annealing lr, compute T_max
                if sc_cfg['target'].split('.')[-1] in ['CosineAnnealingLRWithWarmup', 'CosineAnnealingLR']:
                    T_max = self.max_epochs * self.dataloader['train_iterations']
                    sc_cfg['params']['T_max'] = T_max
                scheduler = instantiate_from_config(sc_cfg)
                op_sc['scheduler'] = {
                    'module': scheduler,
                    'step_iteration': sc_cfg.get('step_iteration', 1)
                }
                if op_sc['scheduler']['step_iteration'] == 'epoch':
                    op_sc['scheduler']['step_iteration'] = self.dataloader['train_iterations']
            optimizer_and_scheduler[op_sc['name']] = op_sc

        return optimizer_and_scheduler

    def _get_lr(self, return_type='str'):

        lrs = {}
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            lr = op_sc['optimizer']['module'].state_dict()['param_groups'][0]['lr']
            lrs[op_sc_n + '_lr'] = round(lr, 10)
        if return_type == 'str':
            lrs = str(lrs)
            lrs = lrs.replace('none', 'lr').replace('{', '').replace('}', '').replace('\'', '')
        elif return_type == 'dict':
            pass
        else:
            raise ValueError('Unknow of return type: {}'.format(return_type))
        return lrs

    def sample(self, batch, phase='train', step_type='iteration'):
        tic = time.time()
        self.logger.log_info('Begin to sample...')
        if self.ema is not None:
            self.ema.modify_to_inference()
            suffix = '_ema'
        else:
            suffix = ''

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        with torch.no_grad():
            if self.debug == False:
                if self.args.amp:
                    with autocast():
                        samples = model.sample(batch=batch, step=self.last_iter)
                else:
                    samples = model.sample(batch=batch, step=self.last_iter)
            else:
                samples = model.sample(batch=batch[0].cuda(), step=self.last_iter)

            step = self.last_iter if step_type == 'iteration' else self.last_epoch
            for k, v in samples.items():
                save_dir = os.path.join(self.image_dir, phase, k)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir,
                                         'e{:010d}_itr{:010d}_rank{}{}'.format(self.last_epoch, self.last_iter % self.dataloader['train_iterations'],
                                                                               get_rank(), suffix))
                if torch.is_tensor(v) and v.dim() == 4 and v.shape[1] in [1, 3]:  # image
                    im = v
                    im = im.to(torch.uint8)
                    self.logger.add_images(
                        tag='{}/{}e_{}itr/{}'.format(phase, self.last_epoch, self.last_iter % self.dataloader['train_iterations'], k), img_tensor=im,
                        global_step=step, dataformats='NCHW')

                    # save images
                    im_grid = make_grid(im)
                    im_grid = im_grid.permute(1, 2, 0).to('cpu').numpy()
                    plt.imshow(im_grid)

                    plt.savefig(save_path + '.jpg')
                    self.logger.log_info('save {} to {}'.format(k, save_path + '.jpg'))
                else:  # may be other values, such as text caption
                    with open(save_path + '.txt', 'a') as f:
                        f.write(str(v) + '\n')
                        f.close()
                    self.logger.log_info('save {} to {}'.format(k, save_path + 'txt'))

        if self.ema is not None:
            self.ema.modify_to_train()

        self.logger.log_info('Sample done, time: {:.2f}'.format(time.time() - tic))

    def step(self, batch, phase='train'):
        loss = {}
        if self.debug == False:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()
        else:
            batch = batch[0].cuda()
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            if phase == 'train':
                # check if this optimizer and scheduler is valid in this iteration and epoch
                if op_sc['start_iteration'] > self.last_iter:
                    continue
                if op_sc['end_iteration'] > 0 and op_sc['end_iteration'] <= self.last_iter:
                    continue
                if op_sc['start_epoch'] > self.last_epoch:
                    continue
                if op_sc['end_epoch'] > 0 and op_sc['end_epoch'] <= self.last_epoch:
                    continue

            input = {
                'batch': batch,
                'return_loss': True,
                'step': self.last_iter,
            }
            if op_sc_n != 'none':
                input['name'] = op_sc_n

            if phase == 'train':
                if self.args.amp:
                    with autocast():
                        output = self.model(**input)
                else:
                    output = self.model(**input)
            else:
                with torch.no_grad():
                    if self.args.amp:
                        with autocast():
                            output = self.model(**input)
                    else:
                        output = self.model(**input)

            if phase == 'train':
                if op_sc['optimizer']['step_iteration'] > 0 and (self.last_iter + 1) % op_sc['optimizer']['step_iteration'] == 0:
                    op_sc['optimizer']['module'].zero_grad()
                    if self.args.amp:
                        self.scaler.scale(output['loss']).backward()
                        if self.clip_grad_norm is not None:
                            self.clip_grad_norm(self.model.parameters())
                        self.scaler.step(op_sc['optimizer']['module'])
                        self.scaler.update()
                    else:
                        output['loss'].backward()
                        if self.clip_grad_norm is not None:
                            self.clip_grad_norm(self.model.parameters())
                        op_sc['optimizer']['module'].step()

                if 'scheduler' in op_sc:
                    if op_sc['scheduler']['step_iteration'] > 0 and (self.last_iter + 1) % op_sc['scheduler']['step_iteration'] == 0:
                        if isinstance(op_sc['scheduler']['module'], STEP_WITH_LOSS_SCHEDULERS):
                            op_sc['scheduler']['module'].step(output.get('loss'))
                        else:
                            op_sc['scheduler']['module'].step()
                # update ema model
                if self.ema is not None:
                    self.ema.update(iteration=self.last_iter)

            loss[op_sc_n] = {k: v for k, v in output.items() if ('loss' in k or 'acc' in k)}
        return loss

    def save(self, force=False):
        if is_primary():
            # save with the epoch specified name
            if self.save_iterations > 0:
                if (self.last_iter + 1) % self.save_iterations == 0:
                    save = True
                else:
                    save = False
            else:
                if isinstance(self.save_epochs, int):
                    save = (self.last_epoch + 1) % self.save_epochs == 0
                else:
                    save = (self.last_epoch + 1) in self.save_epochs

            if save or force:
                state_dict = {
                    'last_epoch': self.last_epoch,
                    'last_iter': self.last_iter,
                    'model': self.model.module.state_dict() if isinstance(self.model,
                                                                          torch.nn.parallel.DistributedDataParallel) else self.model.state_dict()
                }
                if self.ema is not None:
                    state_dict['ema'] = self.ema.state_dict()
                if self.clip_grad_norm is not None:
                    state_dict['clip_grad_norm'] = self.clip_grad_norm.state_dict()

                # add optimizers and schedulers
                optimizer_and_scheduler = {}
                for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
                    state_ = {}
                    for k in op_sc:
                        if k in ['optimizer', 'scheduler']:
                            op_or_sc = {kk: vv for kk, vv in op_sc[k].items() if kk != 'module'}
                            op_or_sc['module'] = op_sc[k]['module'].state_dict()
                            state_[k] = op_or_sc
                        else:
                            state_[k] = op_sc[k]
                    optimizer_and_scheduler[op_sc_n] = state_

                state_dict['optimizer_and_scheduler'] = optimizer_and_scheduler

                if save:
                    save_path = os.path.join(self.ckpt_dir, '{}e_{}iter.pth'.format(str(self.last_epoch).zfill(6), self.last_iter))
                    torch.save(state_dict, save_path)
                    self.logger.log_info('saved in {}'.format(save_path))

                    # save with the last name
                save_path = os.path.join(self.ckpt_dir, 'last.pth')
                torch.save(state_dict, save_path)
                self.logger.log_info('saved in {}'.format(save_path))

    def resume(self,
               path=None,  # The path of last.pth
               load_optimizer_and_scheduler=True,  # whether to load optimizers and scheduler
               load_others=True  # load other informations
               ):
        if path is None:
            path = os.path.join(self.ckpt_dir, 'last.pth')

        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cuda:{}'.format(self.args.local_rank))

            if load_others:
                self.last_epoch = state_dict['last_epoch']
                self.last_iter = state_dict['last_iter']

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                try:
                    self.model.module.load_state_dict(state_dict['model'])  # , strict=False)
                except:
                    model_dict = self.model.module.state_dict()
                    temp_state_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict.keys()}
                    model_dict.update(temp_state_dict)
                    self.model.module.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(state_dict['model'])  # , strict=False)

            if 'ema' in state_dict and self.ema is not None:
                try:
                    self.ema.load_state_dict(state_dict['ema'])
                except:
                    model_dict = self.ema.state_dict()
                    temp_state_dict = {k: v for k, v in state_dict['ema'].items() if k in model_dict.keys()}
                    model_dict.update(temp_state_dict)
                    self.ema.load_state_dict(model_dict)

            if 'clip_grad_norm' in state_dict and self.clip_grad_norm is not None:
                self.clip_grad_norm.load_state_dict(state_dict['clip_grad_norm'])

            # handle optimizer and scheduler
            for op_sc_n, op_sc in state_dict['optimizer_and_scheduler'].items():
                for k in op_sc:
                    if k in ['optimizer', 'scheduler']:
                        for kk in op_sc[k]:
                            if kk == 'module' and load_optimizer_and_scheduler:
                                self.optimizer_and_scheduler[op_sc_n][k][kk].load_state_dict(op_sc[k][kk])
                            elif load_others:  # such as step_iteration, ...
                                self.optimizer_and_scheduler[op_sc_n][k][kk] = op_sc[k][kk]
                    elif load_others:  # such as start_epoch, end_epoch, ....
                        self.optimizer_and_scheduler[op_sc_n][k] = op_sc[k]

            self.logger.log_info('Resume from {}'.format(path))

    def train_epoch(self):
        self.model.train()
        self.last_epoch += 1

        if self.args.distributed:
            self.dataloader['train_loader'].sampler.set_epoch(self.last_epoch)

        epoch_start = time.time()
        itr_start = time.time()
        itr = -1
        for itr, batch in enumerate(self.dataloader['train_loader']):
            if itr == 0:
                print("Time is " + str(time.strftime('%X %x %Z')))
            data_time = time.time() - itr_start
            step_start = time.time()
            self.last_iter += 1
            loss = self.step(batch, phase='train')
            # logging info
            if self.logger is not None and self.last_iter % self.args.log_frequency == 0:
                info = '{}: train'.format(self.args.name)
                info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs,
                                                                self.last_iter % self.dataloader['train_iterations'],
                                                                self.dataloader['train_iterations'])
                for loss_n, loss_dict in loss.items():
                    info += ' ||'
                    loss_dict = reduce_dict(loss_dict)
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    # info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs, self.last_iter%self.dataloader['train_iterations'], self.dataloader['train_iterations'])
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(tag='train/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]), global_step=self.last_iter)

                # log lr
                lrs = self._get_lr(return_type='dict')
                for k in lrs.keys():
                    lr = lrs[k]
                    self.logger.add_scalar(tag='train/{}_lr'.format(k), scalar_value=lrs[k], global_step=self.last_iter)

                # add lr to info
                info += ' || {}'.format(self._get_lr())

                # add time consumption to info
                spend_time = time.time() - self.start_train_time
                itr_time_avg = spend_time / (self.last_iter + 1)
                info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | iter_avg_time: {ita}s | epoch_time: {et} | spend_time: {st} | left_time: {lt}'.format(
                    dt=round(data_time, 1),
                    it=round(time.time() - itr_start, 1),
                    fbt=round(time.time() - step_start, 1),
                    ita=round(itr_time_avg, 1),
                    et=format_seconds(time.time() - epoch_start),
                    st=format_seconds(spend_time),
                    lt=format_seconds(itr_time_avg * self.max_epochs * self.dataloader['train_iterations'] - spend_time)
                )
                self.logger.log_info(info)

            itr_start = time.time()

        # modify here to make sure dataloader['train_iterations'] is correct
        assert itr >= 0, "The data is too less to form one iteration!"
        self.dataloader['train_iterations'] = itr + 1

    def validate_epoch(self):
        if 'validation_loader' not in self.dataloader:
            val = False
        else:
            if isinstance(self.validation_epochs, int):
                val = (self.last_epoch + 1) % self.validation_epochs == 0
            else:
                val = (self.last_epoch + 1) in self.validation_epochs

        if val:
            if self.args.distributed:
                self.dataloader['validation_loader'].sampler.set_epoch(self.last_epoch)

            self.model.eval()
            overall_loss = None
            epoch_start = time.time()
            itr_start = time.time()
            itr = -1
            for itr, batch in enumerate(self.dataloader['validation_loader']):
                data_time = time.time() - itr_start
                step_start = time.time()
                loss = self.step(batch, phase='val')

                for loss_n, loss_dict in loss.items():
                    loss[loss_n] = reduce_dict(loss_dict)
                if overall_loss is None:
                    overall_loss = loss
                else:
                    for loss_n, loss_dict in loss.items():
                        for k, v in loss_dict.items():
                            overall_loss[loss_n][k] = (overall_loss[loss_n][k] * itr + loss[loss_n][k]) / (itr + 1)

                if self.logger is not None and (itr + 1) % self.args.log_frequency == 0:
                    info = '{}: val'.format(self.args.name)
                    info = info + ': Epoch {}/{} | iter {}/{}'.format(self.last_epoch, self.max_epochs, itr, self.dataloader['validation_iterations'])
                    for loss_n, loss_dict in loss.items():
                        info += ' ||'
                        info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                        # info = info + ': Epoch {}/{} | iter {}/{}'.format(self.last_epoch, self.max_epochs, itr, self.dataloader['validation_iterations'])
                        for k in loss_dict:
                            info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))

                    itr_time_avg = (time.time() - epoch_start) / (itr + 1)
                    info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | epoch_time: {et} | left_time: {lt}'.format(
                        dt=round(data_time, 1),
                        fbt=round(time.time() - step_start, 1),
                        it=round(time.time() - itr_start, 1),
                        et=format_seconds(time.time() - epoch_start),
                        lt=format_seconds(itr_time_avg * (self.dataloader['train_iterations'] - itr - 1))
                    )

                    self.logger.log_info(info)
                itr_start = time.time()
            # modify here to make sure dataloader['validation_iterations'] is correct
            assert itr >= 0, "The data is too less to form one iteration!"
            self.dataloader['validation_iterations'] = itr + 1

            if self.logger is not None:
                info = '{}: val'.format(self.args.name)
                for loss_n, loss_dict in overall_loss.items():
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    info += ': Epoch {}/{}'.format(self.last_epoch, self.max_epochs)
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(tag='val/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]), global_step=self.last_epoch)
                self.logger.log_info(info)

    def validate(self):
        self.validation_epoch()

    def train(self):
        start_epoch = self.last_epoch + 1
        self.start_train_time = time.time()
        self.logger.log_info('{}: global rank {}: start training...'.format(self.args.name, self.args.global_rank), check_primary=False)

        for epoch in range(start_epoch, self.max_epochs):
            self.train_epoch()
            self.save(force=True)
            self.validate_epoch()
