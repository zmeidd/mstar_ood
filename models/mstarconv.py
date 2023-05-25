import torch.nn as nn
import torch

import torch.nn as nn

import torch.nn.functional as F
import collections
import torch.nn.init as init


class MstarConvsss(torch.nn.Module):
    def __init__(self):
        super(MstarConvsss, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, 1, 2)
        self.conv3 = torch.nn.Conv2d(32, 64, 6, 1, 2)
        self.conv4 = torch.nn.Conv2d(64, 128, 5, 1, 2)
        self.conv5 = torch.nn.Conv2d(128, 5, 3, 1, 2)
        self.dense1 = torch.nn.Linear(125, 32)
        self.dense2 = torch.nn.Linear(32, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class mstarVGG(nn.Module):
    def __init__(self, vgg_name):
        super(mstarVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class CNN_12(nn.Module):
    def __init__(self, with_bn=False):
        super(CNN_12, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 5)
        self.weight_init()  # initialize weights ourselves
        self.with_bn = False
        if with_bn:
            self.with_bn = True
            self.conv1_bn = nn.BatchNorm2d(16)
            self.conv2_bn = nn.BatchNorm2d(32)
            self.conv3_bn = nn.BatchNorm2d(64)
            self.conv4_bn = nn.BatchNorm2d(128)
            self.conv5_bn = nn.BatchNorm2d(256)
            self.fc1_bn = nn.BatchNorm1d(256)

    def weight_init(self):
        init.xavier_uniform(self.conv1.weight)
        init.constant(self.conv1.bias, 0)
        init.xavier_uniform(self.conv2.weight)
        init.constant(self.conv2.bias, 0)
        init.xavier_uniform(self.conv3.weight)
        init.constant(self.conv3.bias, 0)
        init.xavier_uniform(self.conv4.weight)
        init.constant(self.conv4.bias, 0)
        init.xavier_uniform(self.conv5.weight)
        init.constant(self.conv5.bias, 0)
        init.xavier_uniform(self.fc1.weight)
        init.constant(self.fc1.bias, 0)
        init.xavier_uniform(self.fc2.weight)
        init.constant(self.fc2.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        if self.with_bn:
            x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), 2)
        if self.with_bn:
            x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv3(x), 3, stride=2)
        if self.with_bn:
            x = self.conv3_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv4(x), 2)
        if self.with_bn:
            x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.with_bn:
            x = self.conv5_bn(x)
        x = F.relu(x)
        x = x.view(-1, 256)
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        if self.with_bn:
            x = self.fc1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
