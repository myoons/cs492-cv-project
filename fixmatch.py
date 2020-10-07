import numpy as np
import torch
from torchvision import utils
import torch.nn as nn

from torchvision.transforms import transforms
from augment import rand_augment, weak_augmentation
from utils import *

import random
import math


MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

def weak_aug(resize, imsize):
    weak_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(imsize),
        weak_augmentation.flip_augmentation(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return weak_transform


def strong_aug(resize, imsize):
    transform_train = transforms.Compose([
        transforms.Resize(opts.imResize),
        transforms.RandomResizedCrop(opts.imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Add RandAugment with N, M(hyperparameter)
    return transform_train.transforms.insert(0, rand_augment(N, M))


def learning_decay(n, k, K):
    return n * math.cos((7 * math.PI) / (16 * K))


def train():
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_un = AverageMeter()
    
    losses_curr = AverageMeter()
    losses_x_curr = AverageMeter()
    losses_un_curr = AverageMeter()

    weight_scale = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

def loss(self, output_x, target_x, output_u, target_u, epoch):    
    def idx_f(qb, threshold):
        if max(qb) > threshold:
            return 1
        else:
            return 0
    Ls = -torch.mean(torch.sum(F.log_softmax(output_x, dim) * target_x, dim=1))

    H = - (1-target_u) * np.log(1 - target_u)) - target_u * np.log(prob(strong_aug(ub)))
    Lu = np.avg(idx_f(target_u, self.threshold) * H)

    return Ls, labmda_u * Lu

__call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, final_epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, opts.lambda_u * linear_rampup(epoch, final_epoch)


class FixMatch(nn.Module):
    def __init__(self, imagenet):
        super.__init__()
        self.imagenet = imagenet
        

