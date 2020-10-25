from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import argparse
import shutil
import random
import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

from dataset import get_dataset
from models import Res18, Res34, Res50
from utils import *
from fixmatch import FixMatch

import nsml
from nsml import DATASET_PATH, IS_ON_NSML


NUM_CLASSES = 265
best_acc = 0

def run_session(fixmatch, args, labeled_trainloader, unlabeled_trainloader, validation_loader, global_step):
    global best_acc

    start_epoch = 0
    test_accs = []

    for epoch in range(fixmatch.args.start_epoch, fixmatch.args.epochs):
        loss, loss_x, loss_u, avg_top1, avg_top5 = fixmatch.train(args, labeled_trainloader, unlabeled_trainloader, global_step)

        print('epoch {:03d}/{:03d} finished, loss: {:.3f}, loss_x: {:.3f}, loss_un: {:.3f}, learning_rate: {:.6f}, avg_top1: {:.3f}%, avg_top5: {:.3f}%'.format(epoch, args.epochs, loss, loss_x, loss_u, fixmatch.optimizer.param_groups[0]['lr'], avg_top1, avg_top5))

        acc_top1, acc_top5 = fixmatch.validate(args, validation_loader, epoch)
        is_best = acc_top1 > best_acc
        best_acc = max(acc_top1, best_acc)

        if is_best:
            print('model achieved the best accuracy ({:.3f}%) - saving best checkpoint...'.format(best_acc))
            if IS_ON_NSML:
                nsml.save('fixmatch' + '_best')
                torch.save(fixmatch.model.state_dict(), os.path.join('runs', args.name + '_best'))


        if (epoch + 1) % args.save_epoch == 0:
            if IS_ON_NSML:
                nsml.save(args.name + '_e{}'.format(epoch))
                torch.save(fixmatch.model.state_dict(), os.path.join('runs', args.name + '_best'))

        global_step += 1


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument('--num-labeled', type=int, default=1060,
                        help='number of labeled data')
    parser.add_argument('--num-classes', type=int, default=265,
                        help='number of clasess')                        


    parser.add_argument('--name', default='resnet34', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='model name')
    parser.add_argument('--optim', default='sgd', type=str,
                        choices=['adam', 'sgd', 'adamw', 'yogi'],
                        help='optimizer')

    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
    parser.add_argument('--save_epoch', type=int, default=50, help='saving epoch interval')

    parser.add_argument('--imResize', default=256, type=int, help='')
    parser.add_argument('--imsize', default=224, type=int, help='')

    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')

    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')

    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')

    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--pretrained', default=True, help='import pretrained pk file')    

    args = parser.parse_args()
    global_step = 0

    fixmatch = FixMatch(args)
    fixmatch.model.zero_grad()

    model_to_test = fixmatch.model

    if IS_ON_NSML:
        bind_nsml(model_to_test)
        if args.pause:
            nsml.paused(scope=locals())         

    if args.mode == 'train':
        labeled_trainloader, unlabeled_trainloader, validation_loader = get_dataset(args)
        run_session(fixmatch, args, labeled_trainloader, unlabeled_trainloader, validation_loader, global_step)

if __name__ == "__main__":
    main()