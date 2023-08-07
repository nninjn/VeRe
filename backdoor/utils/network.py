import torch
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import torch.nn.functional as F

class FNN_split_6(nn.Module):
    def __init__(self):
        super(FNN_split_6, self).__init__()
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        output = self.out(x)
        return output

class FNN_split_5(nn.Module):
    def __init__(self):
        super(FNN_split_5, self).__init__()
        self.dense5 = nn.Linear(100, 100)
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        x5 = F.relu(self.dense5(x))
        output = self.out(x5)
        return output

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2, 
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2), 
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dense1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
        )
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):

        conv1_uotput = self.conv1(x)
        drop1_output = self.dropout(conv1_uotput)
        conv2_output = self.conv2(drop1_output)
        drop2_output = self.dropout(conv2_output)
        flatten_output = drop2_output.view(drop2_output.size(0), -1)
        dense1_output = self.dense1(flatten_output)
        dense1_relu_output = self.relu(dense1_output)
        drop3_output = self.dropout(dense1_relu_output)
        output = self.dense2(drop3_output)
        return [flatten_output, dense1_output, dense1_relu_output, output]

class CNN_train(nn.Module):

    def __init__(self):
        super(CNN_train, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dense1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
        )
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):

        conv1_uotput = self.conv1(x)
        drop1_output = self.dropout(conv1_uotput)
        conv2_output = self.conv2(drop1_output)
        drop2_output = self.dropout(conv2_output)
        flatten_output = drop2_output.view(drop2_output.size(0), -1)
        dense1_output = self.dense1(flatten_output)
        dense1_relu_output = self.relu(dense1_output)
        drop3_output = self.dropout(dense1_relu_output)
        output = self.dense2(drop3_output)
        return output

class CNNSpilt0(nn.Module):

    def __init__(self):
        super(CNNSpilt0, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU()
        )
        self.dense2 = nn.Linear(512, 10)


    def forward(self, x):

        dense1_output = self.dense1(x)
        output = self.dense2(dense1_output)
        return output



class CNNSpilt2(nn.Module):

    def __init__(self):
        super(CNNSpilt2, self).__init__()
        self.dense2 = nn.Linear(512, 10)


    def forward(self, x):
        output = self.dense2(x)
        return output

cfg = {
    
    'CNN8': [64, 'M',  256, 'M', 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG13_split1(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split1, self).__init__()
        self.in_channels = 3
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        d1 = F.relu(self.dense1(x))
        d2 = F.relu(self.dense2(d1))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        if feature:
            return out
        else:
            return out

class VGG13_split1_sig(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split1_sig, self).__init__()
        self.in_channels = 3
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        d1 = F.sigmoid(self.dense1(x))
        d2 = F.sigmoid(self.dense2(d1))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        if feature:
            return out
        else:
            return out



class VGG13_split2(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split2, self).__init__()
        self.in_channels = 3
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        d2 = F.relu(self.dense2(x))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        if feature:
            return out
        else:
            return out

class VGG13_split2_sig(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split2_sig, self).__init__()
        self.in_channels = 3
        # self.dense2 = nn.Linear(256, 128)
        # self.classifier = nn.Linear(128, 10)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        d2 = F.sigmoid(self.dense2(x))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        if feature:
            return out
        else:
            return out


class VGG13_split3_sig(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split3_sig, self).__init__()
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        out = self.classifier(x)
        if feature:
            return out
        else:
            return out


class VGG13_split3(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split3, self).__init__()
        self.in_channels = 3
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(x)
        if feature:
            return out
        else:
            return out


class VGG13_dense(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_dense, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:2](f2)
        f3 = self.features3[2:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)
        f5_ = self.features5[0:2](f4)
        f5 = self.features5[2:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        if feature:
            return out
        else:
            return [f5, d1, d2, out]

    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG13_split1_gtsrb(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split1_gtsrb, self).__init__()
        self.in_channels = 3
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        d1 = F.relu(self.dense1(x))
        d2 = F.relu(self.dense2(d1))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        if feature:
            return out
        else:
            return out


class VGG13_split2_gtsrb(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split2_gtsrb, self).__init__()
        self.in_channels = 3
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        d2 = F.relu(self.dense2(x))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        if feature:
            return out
        else:
            return out
        
class VGG13_split3_gtsrb(nn.Module):
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_split3_gtsrb, self).__init__()
        self.in_channels = 3
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(x)
        if feature:
            return out
        else:
            return out


class VGG13_gtsrb(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_gtsrb, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:2](f2)
        f3 = self.features3[2:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)
        f5_ = self.features5[0:2](f4)
        f5 = self.features5[2:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        if feature:
            return out
        else:
            return [f5, d1, d2, out]

    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG13_gtsrb_dense(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_gtsrb_dense, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:2](f2)
        f3 = self.features3[2:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)
        f5_ = self.features5[0:2](f4)
        f5 = self.features5[2:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)

        return out


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG11_dense(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG11'):
        super(VGG11_dense, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:2])
        self.features2 = self._make_layers(cfg[vgg_name][2:4])
        self.features3 = self._make_layers(cfg[vgg_name][4:7])
        self.features4 = self._make_layers(cfg[vgg_name][7:10])
        self.features5 = self._make_layers(cfg[vgg_name][10:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:1](x)
        f1 = self.features1[1:](f1_)
        f2_ = self.features2[0:1](f1)
        f2 = self.features2[1:](f2_)
        f3_ = self.features3[0:2](f2)
        f3 = self.features3[2:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)
        f5_ = self.features5[0:2](f4)
        f5 = self.features5[2:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        if feature:
            return out
        else:
            return [f5, d1, d2, out]


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG13(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG13'):
        super(VGG13, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:2](f2)
        f3 = self.features3[2:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)
        f5_ = self.features5[0:2](f4)
        f5 = self.features5[2:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        return out


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
   

class VGG13_sig(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG13'):
        super(VGG13_sig, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        # self.dense1 = nn.Linear(512, 256)
        # self.dense2 = nn.Linear(256, 128)
        # self.classifier = nn.Linear(128, 10)
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:2](f2)
        f3 = self.features3[2:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)
        f5_ = self.features5[0:2](f4)
        f5 = self.features5[2:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.sigmoid(self.dense1(f5))
        d2 = F.sigmoid(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        if feature:
            return out
        else:
            return [f5, d1, d2, out]
            return out

    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.Sigmoid()]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)   
 
class VGG11_split1(nn.Module):
    def __init__(self, vgg_name='VGG11'):
        super(VGG11_split1, self).__init__()
        self.in_channels = 3
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        d1 = F.relu(self.dense1(x))
        d2 = F.relu(self.dense2(d1))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        return out

class VGG11_split2(nn.Module):
    def __init__(self, vgg_name='VGG11'):
        super(VGG11_split2, self).__init__()
        self.in_channels = 3
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        d2 = F.relu(self.dense2(x))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        if feature:
            return out
        else:
            return out
  
    
class VGG11(nn.Module):  # for ai lancet
    
    def __init__(self, vgg_name='VGG11'):
        super(VGG11, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:2])
        self.features2 = self._make_layers(cfg[vgg_name][2:4])
        self.features3 = self._make_layers(cfg[vgg_name][4:7])
        self.features4 = self._make_layers(cfg[vgg_name][7:10])
        self.features5 = self._make_layers(cfg[vgg_name][10:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 43)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:2](f2)
        f3 = self.features3[2:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)
        f5_ = self.features5[0:2](f4)
        f5 = self.features5[2:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        return out


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
        

class VGG16(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG16'):
        super(VGG16, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:10])
        self.features4 = self._make_layers(cfg[vgg_name][10:14])
        self.features5 = self._make_layers(cfg[vgg_name][14:])
        self.dense1 = nn.Linear(25088, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:3](f2)
        f3 = self.features3[3:](f3_)
        f4_ = self.features4[0:3](f3)
        f4 = self.features4[3:](f4_)
        f5_ = self.features5[0:3](f4)
        f5 = self.features5[3:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        return out


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    


class VGG16_dense(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG16'):
        super(VGG16_dense, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:10])
        self.features4 = self._make_layers(cfg[vgg_name][10:14])
        self.features5 = self._make_layers(cfg[vgg_name][14:])
        self.dense1 = nn.Linear(25088, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:3](f2)
        f3 = self.features3[3:](f3_)
        f4_ = self.features4[0:3](f3)
        f4 = self.features4[3:](f4_)
        f5_ = self.features5[0:3](f4)
        f5 = self.features5[3:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        if feature:
            return out
        else:
            return [f5, d1, d2, out]


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class VGG16_split1(nn.Module):
    def __init__(self, vgg_name='VGG16'):
        super(VGG16_split1, self).__init__()
        self.in_channels = 3
        self.dense1 = nn.Linear(25088, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        d1 = F.relu(self.dense1(x))
        d2 = F.relu(self.dense2(d1))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        if feature:
            return out
        else:
            return out
        

class VGG16_split2(nn.Module):
    def __init__(self, vgg_name='VGG16'):
        super(VGG16_split2, self).__init__()
        self.in_channels = 3
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x):
        d2 = F.relu(self.dense2(x))
        # out = d2.view(d2.size(0), -1)
        out = self.classifier(d2)
        return out

class CNN8(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='CNN8'):
        super(CNN8, self).__init__()
        self.in_channels = 1
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:2])
        self.features2 = self._make_layers(cfg[vgg_name][2:4])
        self.features3 = self._make_layers(cfg[vgg_name][4:6])
        self.features4 = self._make_layers(cfg[vgg_name][6:9])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:1](x)
        f1 = self.features1[1:](f1_)
        f2_ = self.features2[0:1](f1)
        f2 = self.features2[1:](f2_)
        f3_ = self.features3[0:1](f2)
        f3 = self.features3[1:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)

        f4 = f4.view(f4.size(0), -1)
        d1 = F.relu(self.dense1(f4))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        return out


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class CNN8_dense(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='CNN8'):
        super(CNN8_dense, self).__init__()
        self.in_channels = 1
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:2])
        self.features2 = self._make_layers(cfg[vgg_name][2:4])
        self.features3 = self._make_layers(cfg[vgg_name][4:6])
        self.features4 = self._make_layers(cfg[vgg_name][6:9])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:1](x)
        f1 = self.features1[1:](f1_)
        f2_ = self.features2[0:1](f1)
        f2 = self.features2[1:](f2_)
        f3_ = self.features3[0:1](f2)
        f3 = self.features3[1:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)

        f4 = f4.view(f4.size(0), -1)
        d1 = F.relu(self.dense1(f4))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        return [f4, d1, d2, out]


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
