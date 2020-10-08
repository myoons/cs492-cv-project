import math

import numpy as np
import random

import torch
import torch.nn as nn

import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    """Decay Learning Rate"""
    lr = args.lr * math.cos((7 *math.pi * epoch)/(args.epochs*16))
    # lr = args.lr * (0.1 ** (epoch // 30))
    args.lr = lr
    
    for param_group in optimizer.param_groups :
        param_group["lr"] = lr
    
# Set Seed
def set_seed(args) :
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# Loss Function for FixMatch
class SemiLoss(object):
    def __call__(self, args, logits_x, targets_x, logits_u_s, logits_u_w):

        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach_() , dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        
        return Lx, Lu, args.lambda_u

class LabelLoss(object):
    def __call__(self, args, logits_x, targets_x):

        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
        
        return Lx

class UnLabelLoss(object):
    def __call__(self, args, logits_u_s, logits_u_w):
        
        pseudo_label = torch.softmax(logits_u_w.detach_() , dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

        return Lu, args.lambda_u
