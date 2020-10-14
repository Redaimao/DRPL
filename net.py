# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules
from collections import OrderedDict


def swish(x):
    return x * F.sigmoid(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        m['bn1'] = nn.BatchNorm2d(n)
        m['ReLU1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(n, n, k, stride=s, padding=1)
        m['bn2'] = nn.BatchNorm2d(n)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))


    def forward(self, x):
        out = self.group1(x) + x
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, n_residual_blocks):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        #self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(6, 64, 9, stride=1, padding=4)  #9 4
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)


        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(64, 2, 3, stride=1, padding=1) #64,2,3 for pair
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = swish(self.relu1(self.bn1(self.conv1(x))))
        x = swish(self.relu2(self.bn2(self.conv2(x))))
        x = swish(self.relu3(self.bn3(self.conv3(x))))
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = swish(self.relu4(self.bn4(self.conv4(y))))# + x
        x = swish(self.relu5(self.bn5(self.conv5(x))))

        return self.sigmoid(self.conv6(x))

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
