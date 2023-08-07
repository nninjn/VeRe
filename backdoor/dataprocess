import torch
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
import random
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
import torch.nn as nn
import copy
import torchvision.transforms as transforms
import os
from typing import Any, Callable, Optional, Tuple

class Data(Dataset):

    def __init__(self, X, Y):

        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

