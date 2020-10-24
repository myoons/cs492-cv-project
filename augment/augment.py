from torchvision import transforms, utils
import random

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