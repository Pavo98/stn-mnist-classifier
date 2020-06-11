from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import thinplate as tps


class LinearLocNet(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 80)
        self.fc3 = nn.Linear(80, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class CNNLocNet(nn.Module):

    def __init__(self, out_features):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(25*128, 100)
        self.fc2 = nn.Linear(100, out_features)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2)
        x = F.adaptive_max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 5)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class STNSimple(nn.Module):

    def __init__(self, outshape, ctrlshape=(10, 10)):
        super().__init__()
        self.nctrl = ctrlshape[0] * ctrlshape[1]
        self.outshape = outshape
        self.nparam = self.nctrl + 2
        ctrl = tps.uniform_grid(ctrlshape)
        self.register_buffer('target_ctrl', ctrl.view(-1, 2))

        in_features = outshape[0] * outshape[1] * outshape[2]
        self.loc = LinearLocNet(in_features, self.nparam * 2)
        self.loc.fc3.bias.data.zero_()
        self.loc.fc3.weight.data.normal_(0, 1e-3)

    def forward(self, x):
        theta = self.loc(x.view(x.shape[0], -1)).view(-1, self.nparam, 2)
        grid = tps.tps_grid(theta, self.target_ctrl, (x.shape[0], ) + self.outshape)
        return F.grid_sample(x, grid)


class STNtps(nn.Module):

    def __init__(self, outshape, ctrlshape=(10, 10)):
        super().__init__()
        self.nctrl = ctrlshape[0] * ctrlshape[1]
        self.outshape = outshape
        self.nparam = self.nctrl + 2
        ctrl = tps.uniform_grid(ctrlshape)
        self.register_buffer('target_ctrl', ctrl.view(-1, 2))

        self.loc = CNNLocNet(self.nparam * 2)
        self.loc.fc2.weight.data.normal_(0, 1e-3)
        self.loc.fc2.bias.data.zero_()

    def forward(self, x):
        theta = self.loc(x).view(-1, self.nparam, 2)
        grid = tps.tps_grid(theta, self.target_ctrl, (x.shape[0], ) + self.outshape)
        return F.grid_sample(x, grid)


class STNaffine(nn.Module):

    def __init__(self):
        super().__init__()

        self.loc = CNNLocNet(2 * 3)
        self.loc.fc2.weight.data.zero_()
        self.loc.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.loc(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(grid)
