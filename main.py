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

from ImageDataLoader import SimpleImageLoader
from models import Res18, Res50, Dense121, Res18_basic

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

from fixmatch import FixMatch


NUM_CLASSES = 265
best_acc = 0

def run_test(fixmatch, labeled_trainloader, unlabeled_trainloader):
    start_epoch = 0
    test_accs = []

    for epoch in range(fixmatch.args.start_epoch, fixmatch.args.epochs):
        train_loss, train_loss_x, train_loss_u, mask_prob = fixmatch.train(
            labeled_trainloader, unlabeled_trainloader,
            ema_model, scheduler, epoch)

        if fixmatch.args.no_progress:
            logger.info("Epoch {}. train_loss: {:.4f}. train_loss_x: {:.4f}. train_loss_u: {:.4f}."
                        .format(epoch+1, train_loss, train_loss_x, train_loss_u))

        test_loss, test_acc = fixmatch.test(args, test_loader, test_model, epoch)

        test_loss, test_acc = test(args, test_loader, test_model, epoch)

        if fixmatch.args.local_rank in [-1, 0]:
            writer.add_scalar('train/1.train_loss', train_loss, epoch)
            writer.add_scalar('train/2.train_loss_x', train_loss_x, epoch)
            writer.add_scalar('train/3.train_loss_u', train_loss_u, epoch)
            writer.add_scalar('train/4.mask', mask_prob, epoch)
            writer.add_scalar('test/1.test_acc', test_acc, epoch)
            writer.add_scalar('test/2.test_loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if fixmatch.args.local_rank in [-1, 0]:
            model_to_save = model.module if hasattr(model, "module") else model
            if fixmatch.args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, fixmatch.args.out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(
            np.mean(test_accs[-20:])))

    if fimxmatch.args.local_rank in [-1, 0]:
        writer.close()
    
    fixmatch.validate(test_loader)

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--name', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50', 'dense121'],
                        help='model name')
    parser.add_argument('--epochs', default=1024, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--k-img', default=65536, type=int,
                        help='number of labeled examples')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc

    fixmatch = FixMatch(args)
    fixmatch.model.zero_grad()
    train(fixmatch)
