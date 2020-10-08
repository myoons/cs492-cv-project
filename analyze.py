import torch
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

import os
from models.models import MyCNN
from utils.randaugment import RandAugment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
PATH = './kaist0011_fashion_dataset_565\Myoons_best\model\model.pt'
device = torch.device('cpu')
NUM_CLASSES = 265

model = MyCNN(NUM_CLASSES)
bestWeight = torch.load(PATH, map_location=device)
model.load_state_dict(bestWeight)

dataset = torchvision.datasets.ImageFolder(root='./fashion_dataset_testhidden/fashion_data2/train/',
                                        transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        # RandAugment(n=2,m=8),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ]))

validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=10,
                                                shuffle=False
                                                )


def custom_imshow(imgList):

    fig = plt.figure()

    rows = 2
    cols = 5

    for i in range(10):
        img = imgList[i]
        temp = fig.add_subplot(rows, cols, i+1)
        temp.imshow(np.transpose(img, (1, 2, 0)))
        temp.axis('off')
    

    plt.show()


def validation(validation_loader, model):
    
    model.eval()

    with torch.no_grad():

        for batch_idx, data in enumerate(validation_loader):

            inputs, labels = data # Inputs = [5, 3 , 224, 224] / Labels = Not Meaningful

            preds = model(inputs)

            pseudo_label = torch.softmax(preds, dim=-1)
            max_probs, targets = torch.max(pseudo_label, dim=-1) # max_probs : [5] / targets : [5]

            if torch.max(max_probs) >= 0.4 :
                print('max_probs :', max_probs)
                print('targets :', targets)
                custom_imshow(inputs)
                
            else :
                print(torch.max(max_probs))

validation(validation_loader, model)