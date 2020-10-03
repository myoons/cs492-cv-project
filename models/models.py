import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import math
from collections import OrderedDict
import re
import torch.nn.functional as F
######################################################################

def conv5x5(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


######################################################################


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine is not None:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

######################################################################

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):  # 512
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.ReLU()]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

######################################################################
# Define the ResNet18-based Model
######################################################################    
class Res18(nn.Module):

    def __init__(self, class_num):
        super(Res18, self).__init__()
        fea_dim = 256
        model_ft = models.resnet18(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(512, class_num)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea = x.view(x.size(0), -1)
        pred = self.classifier(fea)
        return pred
class Res50(nn.Module):

    def __init__(self, class_num):
        super(Res50, self).__init__()
        fea_dim = 256
        model_ft = models.resnet50(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea = x.view(x.size(0), -1)
        pred = self.classifier(fea)
        return pred

class InceptionModule(nn.Module):

    def __init__(self, inChannels):
        super(InceptionModule, self).__init__()

        self.branch1x1 = conv1x1(inChannels, 16)

        self.branch5x5_1 = conv1x1(inChannels, 16)
        self.branch5x5_2 = conv5x5(16, 32)

        self.branch3x3_1 = conv1x1(inChannels, 16)
        self.branch3x3_2 = conv3x3(16,32)
        self.branch3x3_3 = conv3x3(32,64)

        self.branch_pool = nn.Conv2d(inChannels, 32, kernel_size=1)
    
    def forward(self, x):

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branchPool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branchPool = self.branch_pool(branchPool)

        # 16 + 32 + 64 + 32 = 144
        outputs = [branch1x1, branch3x3, branch5x5, branchPool]

        return torch.cat(outputs, 1)
class MainNet(nn.Module):

    def __init__(self, classNum):
        super(MainNet, self).__init__()

        self.conv1 = conv3x3(3,10)
        self.conv2 = conv3x3(144,30)

        self.incept1 = InceptionModule(inChannels=10)
        self.incept2 = InceptionModule(inChannels=30)

        self.max_pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(451584,classNum)

    def forward(self, x):

        batchSize = x.size(0)

        x = F.relu(self.max_pool(self.conv1(x)))
        x = self.incept1(x)

        x = F.relu(self.max_pool(self.conv2(x)))
        x = self.incept2(x)

        x = x.view(batchSize,-1)
        # print(x.size())
        x = self.fc(x)

        return x

class MyCNN(nn.Module):

    def __init__(self, class_num):
        super(MyCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(40)

        self.conv3 = nn.Conv2d(40, 80, kernel_size=3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(80)

        self.conv4 = nn.Conv2d(80, 160, kernel_size=3, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(160)

        self.conv5 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1)
        self.batch5 = nn.BatchNorm2d(320)

        self.conv6 = nn.Conv2d(320, 640, kernel_size=3, stride=1, padding=1)
        self.batch6 = nn.BatchNorm2d(640)

        self.conv7 = nn.Conv2d(640, 1280, kernel_size=3, stride=1, padding=1)
        self.batch7 = nn.BatchNorm2d(1280)

        self.maxPool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(5760, 2048)
        self.batchfc1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.batchfc2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024,class_num)
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

        self.do = nn.Dropout(p=0.1)

    def forward(self, x):
        
        batchSize = x.size(0)
        # torch.Size([53, 3, 224, 224])
        
        x = self.maxPool(self.relu(self.batch1(self.conv1(x))))
        x = self.maxPool(self.relu(self.batch2(self.conv2(x))))
        x = self.maxPool(self.relu(self.batch3(self.conv3(x))))
        x = self.maxPool(self.relu(self.batch4(self.conv4(x))))
        x = self.maxPool(self.relu(self.batch5(self.conv5(x))))
        x = self.maxPool(self.relu(self.batch6(self.conv6(x))))

        x = x.view(batchSize,-1)
        x = self.do(x)
        x = self.relu(self.batchfc1(self.fc1(x)))
        x = self.relu(self.batchfc2(self.fc2(x)))
        out = self.fc3(x)
        return out