import torch
import torch.nn as nn
import math
from Dataset import *
from torch.utils.data import DataLoader

class CalipsoMLP(nn.Module):
    """!@brief doxygen test

    doxygen test
    """

    def __init__(self, num_classes, channels, density, hidden_layers = [3,3]):
        super(CalipsoMLP, self).__init__()
        #CST05312024
        #self.input_size = channels * density
        self.input_size = (channels+5)*3
        print(self.input_size)
        self.output_size = num_classes
        self.hidden_size = [int(i * self.input_size) for i in hidden_layers]
        #self.num_lag = vario_num_lag

        self.input = nn.Linear(self.input_size, self.hidden_size[0])
        self.lrelu = nn.LeakyReLU()
        #self.lrelu = nn.Tanh()

        self.hidden = nn.ModuleList()

        for i in range(len(hidden_layers) - 1):
            self.hidden.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))

        self.output = nn.Linear(self.hidden_size[-1], self.output_size)

    def forward(self,X):
        # Run directional variogram on input images and reshape for network input
        #print(splitImgs.shape)

        #x = splitImgs.view(splitImgs.shape[0],splitImgs.shape[2],splitImgs.shape[3])
        #x = torch.from_numpy(split_img_vario)
        X = X.view(X.shape[0], -1)
        x = X #CST05312024 this resizes the image causing the NN to crash if using the 3-4-5 Vario function
        # Run forward pass through network on variogram output
        x = x.to(torch.float32)
        x = self.input(x)
        x = self.lrelu(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            x = self.lrelu(x)
        x = self.output(x)
        #x = self.lrelu(x)

        return x

