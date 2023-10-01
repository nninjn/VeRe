import torch
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import torch.nn.functional as F


class FNN(nn.Module):

    def __init__(self):
        super(FNN, self).__init__()

        self.dense1 = nn.Linear(5, 50)
        self.dense2 = nn.Linear(50, 50)
        self.dense3 = nn.Linear(50, 50)
        self.dense4 = nn.Linear(50, 50)
        self.dense5 = nn.Linear(50, 50)
        self.dense6 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 5)



    def forward(self, x):
        x1 = F.relu(self.dense1(x))
        x2 = F.relu(self.dense2(x1))
        x3 = F.relu(self.dense3(x2))
        x4 = F.relu(self.dense4(x3))
        x5 = F.relu(self.dense5(x4))
        x6 = F.relu(self.dense6(x5))
        output = self.out(x6)
        return [x1, x2, x3, x4, x5, x6, output]
        # return output


class FNNspilt5(nn.Module):

    def __init__(self):
        super(FNNspilt5, self).__init__()
        self.dense6 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 5)

    def forward(self, x):
        x6 = F.relu(self.dense6(x))
        output = self.out(x6)
        return output


class FNNspilt6(nn.Module):

    def __init__(self):
        super(FNNspilt6, self).__init__()
        self.out = nn.Linear(50, 5)

    def forward(self, x):
        output = self.out(x)
        return output



class CNNSpilt2(nn.Module):

    def __init__(self):
        super(CNNSpilt2, self).__init__()
        self.dense2 = nn.Linear(512, 10)


    def forward(self, x):
        output = self.dense2(x)
        return output

