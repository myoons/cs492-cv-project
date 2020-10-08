import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import csv

import math
from collections import OrderedDict
import re
import torch.nn.functional as F
######################################################################

def conv3x3(inChannel, outChannel):
    return nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=3, stride=1, padding=1)

def batch(size):
    return nn.BatchNorm2d(size)
    
class MyVGG(nn.Module):

    def __init__(self, class_num):
        super(MyVGG, self).__init__()

        #3 224 128
        self.conv1 = conv3x3(3, 64)
        self.conv2 = conv3x3(64, 64)
        self.batch64 = batch(64)

        #64 112 64
        self.conv3 = conv3x3(64, 128)
        self.conv4 = conv3x3(128, 128)
        self.batch128 = batch(128)

        #64 112 64
        self.conv5 = conv3x3(128, 256)
        self.conv6 = conv3x3(256, 256)
        self.conv7 = conv3x3(256, 256)
        self.batch256 = batch(256)

        #256 28 16
        self.conv8 = conv3x3(256, 512)
        self.conv9 = conv3x3(512, 512)
        self.conv10 = conv3x3(512, 512)
        self.batch512 = batch(512)

        #512 14 8
        self.conv11 = conv3x3(512, 512)
        self.conv12 = conv3x3(512, 512)
        self.conv13 = conv3x3(512, 512)

        self.maxPool = nn.MaxPool2d(2)
        self.avgPool = nn.AvgPool2d(7)
        self.nRelu = nn.LeakyReLU(0.2)

        self.fc = nn.Linear(512, class_num)
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.kaiming_normal_(self.conv9.weight)
        nn.init.kaiming_normal_(self.conv10.weight)
        nn.init.kaiming_normal_(self.conv11.weight)
        nn.init.kaiming_normal_(self.conv12.weight)
        nn.init.kaiming_normal_(self.conv13.weight)

        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        
        batchSize = x.size(0) # torch.Size([53, 3, 224, 224])
        
        x = self.nRelu(self.batch64(self.conv1(x))) # torch.Size([53, 64, 224, 224])
        x = self.maxPool(self.nRelu(self.batch64(self.conv2(x)))) # torch.Size([53, 64, 112, 112])

        x = self.nRelu(self.batch128(self.conv3(x))) # torch.Size([53, 128, 224, 224])
        x = self.maxPool(self.nRelu(self.batch128(self.conv4(x)))) # torch.Size([53, 128, 56, 56])

        x = self.nRelu(self.batch256(self.conv5(x))) # torch.Size([53, 256, 56, 56])
        x = self.nRelu(self.batch256(self.conv6(x))) # torch.Size([53, 256, 56, 56])
        x = self.maxPool(self.nRelu(self.batch256(self.conv7(x)))) # torch.Size([53, 256, 28, 28])

        x = self.nRelu(self.batch512(self.conv8(x))) # torch.Size([53, 512, 56, 56])
        x = self.nRelu(self.batch512(self.conv9(x))) # torch.Size([53, 512, 56, 56])
        x = self.maxPool(self.nRelu(self.batch512(self.conv10(x)))) # torch.Size([53, 512, 14, 14])

        x = self.nRelu(self.batch512(self.conv11(x))) # torch.Size([53, 512, 14, 14])
        x = self.nRelu(self.batch512(self.conv12(x))) # torch.Size([53, 512, 14, 14])
        x = self.maxPool(self.nRelu(self.batch512(self.conv13(x)))) # torch.Size([53, 512, 7, 7])
        
        x = self.avgPool(x)
        x = x.view(batchSize,-1) # torch.Size([512, 4000])
        result = self.fc(x)
        
        return result