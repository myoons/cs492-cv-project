# Author: Yeji Han
# 
# Main function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import argparse
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

from utils.misc import split_ids, bind_nsml, get_cosine_schedule_with_warmup
from utils.data import SimpleImageLoader, TransformFix
from utils.train import SemiLoss

NUM_CLASSES = 265

logger = logging.getLogger(__name__)


def create_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=NUM_CLASSES)
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                        depth=args.model_depth,
                                        width=args.model_width,
                                        num_classes=NUM_CLASSES)

    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))

    return model

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Project 1 Training')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--steps_per_epoch', type=int, default=30, metavar='N', help='number of steps to train per epoch (-1: num_data//batchsize)')

# basic settings
parser.add_argument('--arch', default='wideresnet', type=str, choices=['wideresnet', 'resnext'], help='architecture name')
parser.add_argument('--name',default='WiderResNet', type=str, help='output model name')
parser.add_argument('--gpu_ids',default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--num_workers', type=int, default=3, help='number of workers')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay rate (0: no ema model)')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=50, help='saving epoch interval')

# hyper-parameters for fixmatch
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--mu', default=5, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--T', default=0.75, type=float)
parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml 
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

def main():
    ######################################################################
    # Initiation
    ###################################################################### 
    args = parser.parse_args()
    args.cuda = 0

    global_step = 0

    print(args)
    print(torch.__version__)
    
    # Set GPU
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        args.cuda = 1
        print("Currently using GPU {}".format(args.gpu_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
        args.device = torch.device(0)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        args.device = torch.device('cpu')

    if args.arch == 'wideresnet':
        args.model_depth = 28
        args.model_width = 10

    if args.arch == 'resnext':
        args.model_cardinality = 8
        args.model_depth = 29
        args.model_width = 64

    ######################################################################
    # Create a Model
    ######################################################################     
    model = create_model(args)
    model.to(args.device)

    if len(args.gpu_ids.split(',')) > 1:
        model = torch.nn.DataParallel(model)
    model.train()

    ### DO NOT MODIFY THIS BLOCK ###
    # if IS_ON_NSML:
    #     bind_nsml(model)
    #     if args.pause:
    #         nsml.paused(scope=locals())
    ################################

    ######################################################################
    # Load dataset
    ######################################################################     
    train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)    
    augmentation = TransformFix(args.imResize, args.imsize)
    weak_transform, strong_transform = augmentation()

    print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
    labeled_trainloader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(args.imResize),
                                  transforms.RandomResizedCrop(args.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    unlabeled_weak_trainloader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                            transform=weak_transform),
                            batch_size=args.batchsize*args.mu, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    unlabeled_strong_trainloader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                            transform=strong_transform),
                            batch_size=args.batchsize*args.mu, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imResize),
                                   transforms.CenterCrop(args.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                               batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    ######################################################################
    # Set an optimizer
    ######################################################################     
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.iteration = len(labeled_trainloader) // args.batchsize
    args.total_steps = args.epochs * args.iteration 

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)                          
    if args.steps_per_epoch < 0:
        args.steps_per_epoch = len(labeled_trainloader)

    # INSTANTIATE LOSS CLASS
    train_criterion = SemiLoss()

    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps_per_epoch)

    start_epoch = 0
    # Train and Validation
    #  

def train():
    pass

def test():
    pass


if __name__ == '__main__':
    main()