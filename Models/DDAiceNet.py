import torch
import torch.nn as nn
import math
from utils import directional_vario
from Dataset import *
from torch.utils.data import DataLoader

class DDAiceNet(nn.Module):

	def __init__(self, numClasses, numFeats = 3, hiddenLayers = [3,3]):
		super(DDAiceNet, self).__init__()

		self.input_size = numFeats * 3
		self.output_size = numClasses
		self.hidden_size = [int(i * self.input_size) for i in hiddenLayers]

		self.input = nn.Linear(self.input_size, self.hidden_size[0])
		self.lrelu = nn.LeakyReLU()
		self.output = nn.Linear(self.hidden_size[-1], self.output_size)

		self.hidden = nn.ModuleList()
		for i in range(len(hidden_layers) - 1):
            self.hidden.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))

	def forward(self, alongTrackChunk):

		x = alongTrackChunk.view(alongTrackChunk.shape[0], -1)

		# Run forward pass through network on ...
        x = self.input(x)
        x = self.lrelu(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            x = self.lrelu(x)
        x = self.output(x)