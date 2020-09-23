from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import numpy as np
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU,QuantLinear

class QuantLeNet(nn.Module):
    def __init__(self,weight_bit_width=4,acti_bit_width=8):
        super(QuantLeNet, self).__init__()
        self.conv1 = QuantConv2d(1, 6, 5, padding=2,weight_bit_width=weight_bit_width)
        # self.relu1 = QuantReLU(bit_width=acti_bit_width, max_val=6)
        self.conv2 = QuantConv2d(6, 16, 5, weight_bit_width=weight_bit_width)
        # self.relu2 = QuantReLU(bit_width=acti_bit_width, max_val=6)
        self.fc1   = QuantLinear(16*5*5, 120, bias=True, weight_bit_width=weight_bit_width)
        # self.relu3 = QuantReLU(bit_width=acti_bit_width, max_val=6)
        self.fc2   = QuantLinear(120, 84, bias=True, weight_bit_width=weight_bit_width)
        # self.relu4 = QuantReLU(bit_width=acti_bit_width, max_val=6)
        self.fc3   = QuantLinear(84, 10, bias=True, weight_bit_width=weight_bit_width)
        # self.relu5 = QuantReLU(bit_width=acti_bit_width, max_val=6)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)
class LeNet(nn.Module):

    # original network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)
