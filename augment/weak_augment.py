from torchvision import transforms
import random

def flip_augmentation():
    return transforms.RandomHorizontalFlip(p=0.5)

def affine_augmentation():
    shift_horizontal = random.randrange(0, 125) / 1000
    shift_vertical = random.randrange(0, 125) / 1000
    return transforms.RandomAffine(0, translate=(shift_horizontal, shift_vertical), scale=None, shear=None, resample=False, fillcolor=0)