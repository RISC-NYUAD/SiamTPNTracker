import os
import sys
sys.path.append('./')
import argparse
import time
import importlib
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import random
import numpy as np
import lib.train.admin.settings as ws_settings
from lib.train.trainers import LTRTrainer
from lib.train.actors import build_actor
from lib.config.default import cfg, update_config_from_file
from lib.train.base_functions import update_settings, names2datasets, get_optimizer_scheduler, build_dataloaders
#from lib.models import model_factory 
from lib.models.siamtpn.track import build_network

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None):
    """Run the train script.
    args:
        config_name: Name of the yaml file in the "experiments/<script_name>".
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """
    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")
    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('config_name: {}.yaml'.format(config_name))

    '''2021.1.5 set seed for different process'''
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    settings = ws_settings.Settings()
    settings.description = 'Training script for object tracking'
    settings.config_name = config_name
    settings.project_path = 'train/{}'.format(config_name)
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    settings.cfg_file = 'experiments/{}.yaml'.format(config_name)

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    update_config_from_file(cfg, settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "{}-{}.log".format(settings.config_name, int(time.time())))

    # Build dataloaders
    loader_train = build_dataloaders(cfg, settings)

    # Create network
    net = build_network(cfg)
    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    # Loss functions and Actors

    actor = build_actor(cfg, net)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)



def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--config', type=str, required=True, help="Name of the config file.")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--save_dir', type=str, default='./results', help='the directory to save checkpoints and logs')
    parser.add_argument('--seed', type=int, default=40, help='seed for random numbers')
    args = parser.parse_args()
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    run_training(args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed)


if __name__ == '__main__':
    main()
