import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torchvision.transforms import transforms

import os
import random
import math
import time

from ema import ModelEMA
from utils import *
from models import Res18, Res50, Dense121, Res18_basic

from nsml import IS_ON_NSML


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class FixMatch(object):
    def __init__(self, args):
        self.args = args

        if args.name == 'resnet18':
            self.model = Res18(args.num_classes)
        elif args.name == 'resnet50':
            self.model = Res50(args.num_classes)
        elif args.name == 'dense121':
            self.model = DenseNet(args.num_classes)
        else:
            self.model = Res18_basic(args.num_classes)

        print(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = device        
        print('Using device:', device)
        print()

        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

            self.model = self.model.to(args.device)

        if args.local_rank == -1:
            device = torch.device('cuda', args.gpu_id)
            args.world_size = 1
            args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda', args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            args.world_size = torch.distributed.get_world_size()
            args.n_gpu = 1
     
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        if args.seed != -1:
            set_seed(args)
        
        if args.optim == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                                        weight_decay=args.wdecay)
        else:    
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                        momentum=0.9, nesterov=args.nesterov)
        args.iteration = args.num_labeled // args.batch_size // args.world_size
        args.total_steps = args.epochs * args.iteration
        args.warmup = args.warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, args.warmup * args.iteration, args.total_steps)

        if args.use_ema:
            self.ema_model = ModelEMA(args, self.model, args.ema_decay, args.device)

        if args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[args.local_rank],
                output_device=args.local_rank, 
                find_unused_parameters=True)

    def train(self, args, labeled_trainloader, unlabeled_w_trainloader, unlabeled_s_trainloader, global_step):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        
        losses_curr = AverageMeter()
        losses_x_curr = AverageMeter()
        losses_u_curr = AverageMeter()

        weight_scale = AverageMeter()
        acc_top1 = AverageMeter()
        acc_top5 = AverageMeter()
        end = time.time()

        train_loader = zip(labeled_trainloader, unlabeled_w_trainloader, unlabeled_s_trainloader)
        self.model.train()

        for batch_idx, (data_x, data_w_u, data_s_u) in enumerate(train_loader):
            input_x, target_x = data_x
            input_u_w, input_u_s = data_w_u[0], data_s_u[0]

            data_time.update(time.time())
            batch_size = input_x.shape[0]

            if args.device.type == 'cuda':
                input_x = input_x.to(args.device)
                input_u_w, input_u_s = input_u_w.to(args.device), input_u_s.to(args.device)            
                target_x = target_x.to(args.device)

            targets_org = target_x

            emb_x, logits_x = self.model(input_x)
            emb_u_w, logits_u_w = self.model(input_u_w)
            emb_u_s, logits_u_s = self.model(input_u_s)

            Lx = F.cross_entropy(logits_x, target_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, target_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, target_u, reduction='none') * mask).mean()

            if global_step < 20:
                loss = Lx
            else:            
                loss = Lx + args.lambda_u * Lu
            loss.backward()

            losses_curr.update(loss.item(), input_x.size(0))
            losses_x_curr.update(Lx.item(), input_x.size(0))
            losses_u_curr.update(Lu.item(), input_x.size(0))

            losses.update(loss.item(), input_x.size(0))
            losses_x.update(Lx.item(), input_x.size(0))
            losses_u.update(Lu.item(), input_x.size(0))
            
            self.optimizer.step()
            self.scheduler.step()

            if args.use_ema:
                self.ema_model.update(self.model)

            self.model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_prob = mask.mean().item()

            with torch.no_grad():
                    # compute guessed labels of unlabel samples
                embed_x, pred_x1 = self.model(input_x)

                if IS_ON_NSML and global_step % args.log_interval == 0:
                    nsml.report(step=global_step, loss=losses_curr.avg, loss_x=losses_x_curr.avg, loss_un=losses_u_curr.avg)
                    losses_curr.reset()
                    losses_x_curr.reset()
                    losses_u_curr.reset()

                acc_top1b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=1)*100
                acc_top5b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=5)*100    
                acc_top1.update(torch.as_tensor(acc_top1b), input_x.size(0))        
                acc_top5.update(torch.as_tensor(acc_top5b), input_x.size(0))


        return losses.avg, losses_x.avg, losses_u.avg, acc_top1.avg, acc_top5.avg


    def validate(self, args, test_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                data_time.update(time.time() - end)
                self.model.eval()

                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                emb_o, outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                prec1 = top_n_accuracy_score(targets.data.cpu().numpy(), outputs.data.cpu().numpy(), n=1)*100
                prec5 = top_n_accuracy_score(targets.data.cpu().numpy(), outputs.data.cpu().numpy(), n=5)*100    

                losses.update(loss.item(), inputs.shape[0])
                top1.update(prec1, inputs.shape[0])
                top5.update(prec5, inputs.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()

        print("top-1 acc: {:.2f}".format(top1.avg))
        print("top-5 acc: {:.2f}".format(top5.avg))

        return top1.avg, top5.avg