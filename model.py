import numpy as np
import torch
from torchvision import transforms, utils

import random
import math

def weak_aug(xb):
    flip = transforms.RandomHorizontalFlip(p=0.5)
    shift_horizontal = random.randrange(0, 125) / 1000
    shift_vertical = random.randrange(0, 125) / 1000
    shift = transforms.RandomAffine(0, translate=(shift_horizontal, shift_vertical), scale=None, shear=None, resample=False, fillcolor=0)
    aug = transforms.Compose([
        flip,
        shift,])
    return aug(xb)

def rand_augment(ub, M):
    M = random.randrange(0, 14)
    transformations = []
    choice_idx = random.randrange(0, len(transformations))
    aug = transformations[choice_idx]
    pass

def ct_augment(ub):
    magnitude_bins = []
    aug1 = magnitude_bin * transformation1
    aug2 = magnitude_bin * transformation2
    
    result1 = aug1(ub)
    result2 = aug2(ub)

    pass

def cutout():
    pass

def learning_decay(n, k, K):
    return n * math.cos((7 * math.PI) / (16 * K))

def loss_function(prob, weak_aug, strong_aug, xb, pb, ub, threshold, labmda_u):
    def idx_f(qb, threshold):
        if max(qb) > threshold:
            return 1
        else:
            return 0

    pm = prob(weak_aug(xb))
    H = - (1- pb) * np.log(1 - pm) - pb * np.log(pm)
    ls = np.avg(H)

    qb = prob(weak_aug(ub))
    qb_hat = np.argmax(qb)
    H = - (1- qb_hat) * np.log(1 - prob(strong_aug(ub))) - qb_hat * np.log(prob(strong_aug(ub)))
    lu = np.avg(idx_f(qb, threshold) * H)

    return ls + labmda_u * lu
