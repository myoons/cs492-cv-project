import torch
import shutil
import nsml
import numpy as np
import os

from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms
import time
import random
import models

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
    
def default_image_loader(path):
    return Image.open(path).convert('RGB')


class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, loader=default_image_loader, strong_transform=None):
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
        self.strong_transform = strong_transform
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
        if self.split != 'unlabel':
            if self.transform is not None:
                img = self.transform(img)
            label = self.imclasses[index]
            return img, label
        else:        
            img1, img2 = self.transform(img), self.strong_transform(img)
            return img1, img2
        
    def __len__(self):
        return len(self.imnames)


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


### NSML functions
def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(root_path, 'test',
                               transform=transforms.Compose([
                                   transforms.Resize(224),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), 
                               batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
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


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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

def load_best_model(path, num_classes, args):    
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    model = models.Res34(num_classes)
    bestWeight = torch.load(path, map_location=args.device)
    model.load_state_dict(bestWeight)

    return model