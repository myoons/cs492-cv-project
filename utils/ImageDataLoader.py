from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

import torch
import torchvision

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transformWeak=None, transformStrong=None, loader=default_image_loader):
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

        self.transformWeak = transformWeak
        self.transformStrong = transformStrong
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
    
    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))
        
        if self.split == 'test': # Test
            if self.transformWeak is not None:
                img = self.transformWeak(img)
            return img
        elif self.split != 'unlabel': # Labeled
            if self.transformWeak is not None:
                img = self.transformWeak(img)
            label = self.imclasses[index]

            return img, label
        else:  # Unlabeled

            weakAugmented = self.transformWeak(img)
            strongAugmented = self.transformStrong(img)
            # weakAugmentedTwo = self.transformWeak(img)

            return weakAugmented, strongAugmented
        
    def __len__(self):
        return len(self.imnames)
