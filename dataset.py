import os

from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms

from utils import split_ids

from nsml import DATASET_PATH, IS_ON_NSML
from augment.rand_augment import RandAugment

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

N, M = 16, 30

def default_image_loader(path):
    return Image.open(path).convert('RGB')


def strong_aug(resize, imsize):
    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Add RandAugment with N, M(hyperparameter)
    transform_train.transforms.insert(0, RandAugment(N, M))
    return transform_train


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, loader=default_image_loader):
        if split == 'test':
            self.impath = os.path.join(rootdir, 'test_data')
            meta_file = os.path.join(self.impath, 'test_meta.txt')
        else:
            self.impath = os.path.join(rootdir, 'train/train_data')
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []
        
        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()        
                if int(label) == -1 and (split != 'unlabel' and split != 'test'):
                    continue
                if int(label) != -1 and (split == 'unlabel' or split == 'test'):
                    continue
                if (ids is None) or (int(instance_id) in ids):
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        if split == 'train' or split == 'val':
                            imclasses.append(int(label))

        self.transform = transform
        self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
    
    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))
        
        if self.split == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.split != 'unlabel':
            if self.transform is not None:
                img = self.transform(img)
            label = self.imclasses[index]
            return img, label
        else:        
            img1, img2 = self.TransformTwice(img)
            return img1, img2
        
    def __len__(self):
        return len(self.imnames)


def get_dataset(args):
    train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
    print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
    labeled_trainloader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                            transform=transforms.Compose([
                                transforms.Resize(args.imResize),
                                transforms.RandomResizedCrop(args.imsize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=MEAN, std=STD),])),
                            batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    print('train_loader done')

    unlabeled_w_trainloader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                            transform=transforms.Compose([
                                transforms.Resize(args.imResize),
                                transforms.RandomResizedCrop(args.imsize),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=MEAN, std=STD),])),
                            batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    unlabeled_s_trainloader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                            transform=strong_aug(args.imResize, args.imsize)),
                            batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        
    print('unlabel_loader done')

    validation_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                            transform=transforms.Compose([
                                transforms.Resize(args.imResize),
                                transforms.CenterCrop(args.imsize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=MEAN, std=STD),])),
                            batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    print('validation_loader done')


    return labeled_trainloader, unlabeled_w_trainloader, unlabeled_s_trainloader, validation_loader

train_ids, val_ids, unl_ids = split_ids(os.path.join('fashion_data2/', 'train/train_label'), 0.2)
print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
labeled_trainloader = torch.utils.data.DataLoader(
    SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                        transform=transforms.Compose([
                            transforms.Resize(args.imResize),
                            transforms.RandomResizedCrop(args.imsize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=MEAN, std=STD),])),
                        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
print('train_loader done')

print(type(labeled_trainloader))

# unlabeled_w_trainloader = torch.utils.data.DataLoader(
#     SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
#                         transform=transforms.Compose([
#                             transforms.Resize(args.imResize),
#                             transforms.RandomResizedCrop(args.imsize),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.RandomVerticalFlip(),
#                             transforms.ToTensor(),
#                             transforms.Normalize(mean=MEAN, std=STD),])),
#                         batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

# unlabeled_s_trainloader = torch.utils.data.DataLoader(
#     SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
#                         transform=strong_aug(args.imResize, args.imsize)),
#                         batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
# print('unlabel_loader done')

# validation_loader = torch.utils.data.DataLoader(
#     SimpleImageLoader(DATASET_PATH, 'val', val_ids,
#                         transform=transforms.Compose([
#                             transforms.Resize(args.imResize),
#                             transforms.CenterCrop(args.imsize),
#                             transforms.ToTensor(),
#                             transforms.Normalize(mean=MEAN, std=STD),])),
#                         batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
# print('validation_loader done')


# return labeled_trainloader, unlabeled_w_trainloader, unlabeled_s_trainloader, validation_loader

