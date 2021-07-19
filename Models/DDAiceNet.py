import torch
import torch.nn as nn
import math
from utils import directional_vario
from Dataset import *
from torch.utils.data import DataLoader

class DDAiceNet(nn.Module):

	def __init__(self, numClasses, hiddenLayers = [3,3]):
		super(DDAiceNet, self).__init__()

		self.input_size = 47
		self.output_size = numClasses
		self.hidden_size = [int(i * self.input_size) for i in hiddenLayers]

		self.input = nn.Linear(self.input_size, self.hidden_size[0])
		self.lrelu = nn.LeakyReLU()
		self.output = nn.Linear(self.hidden_size[-1], self.output_size)

		self.hidden = nn.ModuleList()
		for i in range(len(hiddenLayers) - 1):
			self.hidden.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))

	def forward(self, alongTrackChunk):

		x = alongTrackChunk.view(alongTrackChunk.shape[0], -1)

		# Run forward pass through network on ...
		print(x)
		x = self.input(x)
		print(x)
		x = self.lrelu(x)
		print(x)
		for i in range(len(self.hidden)):
			x = self.hidden[i](x)
			x = self.lrelu(x)
		print(x)
		x = self.output(x)
		return x