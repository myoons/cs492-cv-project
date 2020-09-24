from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import os
import sys
import argparse

import numpy as np
import shutil
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, models, transforms

import torch.nn.functional as F

from utils.Transform import TransformFix
from utils.ImageDataLoader import SimpleImageLoader
from utils.preTrain import set_seed, SemiLoss, LabelLoss, UnLabelLoss ,adjust_learning_rate, AverageMeter

from models.models import Res50

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

NUM_CLASSES = 265

def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter
        
# For Loading 
def split_ids(path, ratio):
    with open(path) as f:
        ids_l = []
        ids_u = []
        for i, line in enumerate(f.readlines()):
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l.append(int(line[0]))
            else:
                ids_u.append(int(line[0]))

    ids_l = np.array(ids_l)
    ids_u = np.array(ids_u)

    perm = np.random.permutation(np.arange(len(ids_l)))
    cut = int(ratio*len(ids_l))
    train_ids = ids_l[perm][cut:]
    val_ids = ids_l[perm][:cut]

    return train_ids, val_ids, ids_u

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Pytorch FixMatch Fashion Dataset Classify Alogrithm')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--save_epoch', type=int, default=60, metavar='N', help='epoch per saving (default: 1)')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of epochs to train (default: 300)')

# basic settings
parser.add_argument('--name',default='Res', type=str, help='output model name')
parser.add_argument('--gpu_ids',default=0, type=int ,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--n_gpu',default=0, type=int,help='number of gpus using')
parser.add_argument('--batchsize', default=20, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--weightDecay', type=float, default=4e-4, metavar='WD', help='weight decay')
parser.add_argument('--imResize', default=256, type=int, help='Img Resize')
parser.add_argument('--imsize', default=224, type=int, help='Img Crop Size')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')

# hyper-parameters for Fix-Match
parser.add_argument('--lambda_u', default=1, type=float, help='Coefficient of Unlabeled Loss')
parser.add_argument('--threshold', default=0.95, type=float, help='Pseudo Label Threshold')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml 
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

### NSML functions
def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(root_path, 'test',
                               transform=transforms.Compose([
                                   transforms.Resize(args.imResize),
                                   transforms.CenterCrop(args.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=True)
        print('loaded {} test images'.format(len(test_loader.dataset)))

    outputs = []
    s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, probs = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs

def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = model.state_dict()
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)

def main():
    global args, global_step
    args = parser.parse_args()
    global_step = 0

    print("args :",args)

    use_gpu = torch.cuda.is_available()

    if use_gpu :
        device = torch.device('cuda', args.gpu_ids)
        args.device = device
        args.n_gpu = torch.cuda.device_count()
    else :
        device= torch.device('cpu')
        args.device = device

    # Set Seed
    set_seed(args)

    # Set model
    model = Res50(NUM_CLASSES)
    model.to(args.device)
    model.eval()

    # If GPU is available
    if use_gpu :
        print("Currently using GPU {}".format(args.gpu_ids))
        print("Number of GPUs Using",args.n_gpu)
        cudnn.benchmark = True

        # Set multi-gpu (Data Parallel)
        if args.n_gpu > 1:
            model = nn.DataParallel(model)

    else :
        print("Currently using CPU (GPU is highly recommended)")
    
    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model)
        if args.pause:
            nsml.paused(scope=locals())
    ################################

    if args.mode == 'train':

        # Train Model
        model.train()

        # Set dataloader
        train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
        augmentation = TransformFix(args.imResize, args.imsize)
        WA, SA = augmentation()

        print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
        label_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(args.imResize),
                                  transforms.RandomResizedCrop(args.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        print('label_loader done')

        unlabel_weak_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                            transform=WA),
                            batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        
        unlabel_strong_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                            transform=SA),
                            batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

        print('unlabel_loader done')    

        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                               transform=transforms.Compose([
                                  transforms.Resize(args.imResize),
                                  transforms.RandomResizedCrop(args.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                               batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        print('validation_loader done')

        # Set optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay= args.weightDecay, momentum=args.momentum, nesterov=args.nesterov)

        # INSTANTIATE LOSS CLASS
        label_criterion = LabelLoss()
        unlabel_criterion = UnLabelLoss()

        # Train and Validation 
        best_acc = -1
        print('start training')

        model.zero_grad()
        
        for epoch in range(args.start_epoch, args.epochs + 1):

            loss, loss_x, loss_u, avg_top1, avg_top5 = train(args, label_loader,  unlabel_weak_loader, unlabel_strong_loader, model, label_criterion, unlabel_criterion, optimizer, epoch, use_gpu)
            print('epoch {:03d}/{:03d} finished, learning_rate : {}, loss: {:.3f}, loss_x: {:.3f}, loss_un: {:.3f}, avg_top1: {:.3f}%, avg_top5: {:.3f}%'.format(epoch, args.epochs, optimizer.param_groups[0]["lr"], loss, loss_x, loss_u, avg_top1, avg_top5))

            
            acc_top1, acc_top5 = validation(args, validation_loader, model, epoch, use_gpu)
            is_best = acc_top1 > best_acc
            best_acc = max(acc_top1, best_acc)
            if is_best:
                print('model achieved the best accuracy ({:.3f}%) - saving best checkpoint...'.format(best_acc))
                if IS_ON_NSML:
                    nsml.save(args.name + '_best')
                    counter = 0
                else:
                    print(args.name + '_Not In NSML{}'.format(epoch))
            
            
            if (epoch + 1) % args.save_epoch == 0:
                if IS_ON_NSML:
                    nsml.save(args.name + '_e{}'.format(epoch))
                else:
                    print(args.name + '_Not In NSML{}'.format(epoch))
            

            adjust_learning_rate(args, optimizer, epoch)
    

def train(args, label_loader, unlabel_weak_loader, unlabel_strong_loader, model, label_criterion, unlabel_criterion, optimizer, epoch, use_gpu):
    global global_step

    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_un = AverageMeter()
    
    losses_curr = AverageMeter()
    losses_x_curr = AverageMeter()
    losses_un_curr = AverageMeter()

    weight_scale = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    model.train()

    unlabel_loader = zip(unlabel_weak_loader, unlabel_strong_loader)

    for label_idx, data_x in enumerate(label_loader):
        
        optimizer.zero_grad()

        inputs_x, targets_x = data_x
        batchSize = inputs_x.shape[0]
        targets_org = targets_x

        targets_x = targets_x.to(args.device)
        inputs_x = inputs_x.to(args.device)

        fea_x, logits_x = model(inputs_x)

        loss_x = label_criterion(args, logits_x, targets_x)

        for i in range(args.mu) :
            
            loss_un = 0

            for unlabel_idx, (data_u_w, data_u_s) in enumerate(unlabel_loader):
                
                print(unlabel_idx, data_u_w)

                _, inputs_u_w = data_u_w
                _, inputs_u_s = data_u_s

                inputs_u_w = inputs_u_w.to(args.device)
                inputs_u_s = inputs_u_s.to(args.device)

                fea_u_w, targets_u = model(inputs_u_w)
                fea_u_s, logits_u = model(inputs_u_s)

                tempLoss_un = unlabel_criterion(args, logits_u, targets_u)
                loss_un += tempLoss_un

        loss = loss_x + args.lambda_u * (loss_un/args.mu)
            
        losses.update(loss.item(), batchSize)
        losses_x.update(loss_x.item(), batchSize)
        losses_un.update(loss_un.item(), batchSize)

        losses_curr.update(loss.item(), batchSize)
        losses_x_curr.update(loss_x.item(), batchSize)
        losses_un_curr.update(loss_un.item(), batchSize)

        loss.backward()
        optimizer.step()
            
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            embed_x, pred_x1 = model(inputs_x) 

        if IS_ON_NSML and global_step % args.log_interval == 0:
            nsml.report(step=global_step, loss=losses_curr.avg, loss_x=losses_x_curr.avg, loss_un=losses_un_curr.avg)
            losses_curr.reset()
            losses_x_curr.reset()
            losses_un_curr.reset()

        acc_top1b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=1)*100
        acc_top5b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=5)*100    
        acc_top1.update(torch.as_tensor(acc_top1b), batchSize)        
        acc_top5.update(torch.as_tensor(acc_top5b), batchSize)   

        global_step += 1
        
    return losses.avg, losses_x.avg, losses_un.avg, acc_top1.avg, acc_top5.avg

def validation(args, validation_loader, model, epoch, use_gpu):
    model.eval()
    avg_top1= 0.0
    avg_top5 = 0.0
    nCnt =0 
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu :
                inputs = inputs.cuda()
            nCnt +=1
            embed_fea, preds = model(inputs)

            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1)*100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5)*100
            avg_top1 += acc_top1
            avg_top5 += acc_top5

        avg_top1 = float(avg_top1/nCnt)   
        avg_top5= float(avg_top5/nCnt)   
    
    if IS_ON_NSML:
        nsml.report(step=epoch, avg_top1=avg_top1, avg_top5=avg_top5)

    return avg_top1, avg_top5

if __name__ == '__main__':
    main()
