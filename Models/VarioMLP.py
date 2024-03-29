import torch
import torch.nn as nn
import math
from utils import directional_vario
from Dataset import *
from torch.utils.data import DataLoader

class VarioMLP(nn.Module):
    """!@brief doxygen test

    doxygen test
    """

    def __init__(self, num_classes, vario_num_lag, hidden_layers = [3,3]):
        super(VarioMLP, self).__init__()

        self.input_size = vario_num_lag * 3
        self.output_size = num_classes
        self.hidden_size = [int(i * self.input_size) for i in hidden_layers]
        self.num_lag = vario_num_lag

        self.input = nn.Linear(self.input_size, self.hidden_size[0])
        self.lrelu = nn.LeakyReLU()
        #self.lrelu = nn.Tanh()

        self.hidden = nn.ModuleList()

        for i in range(len(hidden_layers) - 1):
            self.hidden.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))

        self.output = nn.Linear(self.hidden_size[-1], self.output_size)

    def forward(self, split_img_vario):

        # Run directional variogram on input images and reshape for network input
        #print(splitImgs.shape)

        #x = splitImgs.view(splitImgs.shape[0],splitImgs.shape[2],splitImgs.shape[3])
        #x = torch.from_numpy(split_img_vario)
        x = split_img_vario.view(split_img_vario.shape[0], -1)

        # Run forward pass through network on variogram output
        x = self.input(x)
        x = self.lrelu(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            x = self.lrelu(x)
        x = self.output(x)
        #x = self.lrelu(x)

        return x
