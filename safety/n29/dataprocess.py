import torch
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import os
from typing import Any, Callable, Optional, Tuple

# BATCH_SIZE = 128
# EPOCH = 10
# LEANR_RATE = 0.001
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Data(Dataset):

    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

