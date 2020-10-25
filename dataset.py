import os

from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms

from utils import split_ids, SimpleImageLoader

from nsml import DATASET_PATH, IS_ON_NSML
from rand_augment import *

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}

N, M = 3, 16

def strong_augmentation(resize, imsize):
    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
        transforms.ToTensor(),
        Lighting(0.1, PCA['eigval'], PCA['eigvec']),
        transforms.Normalize(MEAN, STD),
    ])

    # Add RandAugment with N, M(hyperparameter)
    transform_train.transforms.insert(0, RandAugment(N, M))
    return transform_train

def get_dataset(args):
    train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.1)
    print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
    labeled_trainloader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                            transform=transforms.Compose([
                                transforms.Resize(args.imResize),
                                transforms.RandomResizedCrop(args.imsize),
                                transforms.ColorJitter(
                                    brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.4,
                                ),
                                transforms.ToTensor(),
                                Lighting(0.1, PCA['eigval'], PCA['eigvec']),
                                transforms.Normalize(mean=MEAN, std=STD),]),
                            strong_transform=strong_augmentation(args.imResize, args.imsize)),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print('train_loader done')

    unlabeled_trainloader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                            transform=transforms.Compose([
                                transforms.Resize(args.imResize),
                                transforms.CenterCrop(args.imsize),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ColorJitter(
                                    brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.4,
                                ),
                                transforms.ToTensor(),
                                Lighting(0.1, PCA['eigval'], PCA['eigvec']),
                                transforms.Normalize(mean=MEAN, std=STD),]), 
                            strong_transform=strong_augmentation(args.imResize, args.imsize)),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                            pin_memory=True, drop_last=True)
    
    print('unlabel_loader done')

    validation_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                            transform=transforms.Compose([
                                transforms.Resize(args.imResize),
                                transforms.CenterCrop(args.imsize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=MEAN, std=STD),]),
                            strong_transform=strong_augmentation(args.imResize, args.imsize)),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print('validation_loader done')

    return labeled_trainloader, unlabeled_trainloader, validation_loader