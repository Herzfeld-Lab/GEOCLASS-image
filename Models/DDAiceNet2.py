import torch.nn as nn
from Dataset import *
from torch.utils.data import DataLoader

class DDAiceNet2(nn.Module):

	def __init__(self, numClasses, inSize, hiddenLayers = [3,3]):
		super(DDAiceNet2, self).__init__()

		# convolution stuff...
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3), padding='same')
		self.mp1 = nn.MaxPool2d(kernel_size=3, stride=1)

		self.output = nn.LazyLinear(numClasses)

	def forward(self, varioTensor):
		# print('Model')
		# print(varioTensor.shape)
		# x = varioTensor.view(varioTensor.shape[0], varioTensor.shape[2], varioTensor.shape[3])

		# Run forward pass through network
		x = self.conv1(varioTensor)
		# print(x.shape)
		x = self.mp1(x)
		# print(x.shape)
		x = torch.flatten(x,1)
		# print(x.shape)
		x = self.output(x)
		# print(x.shape)
		return x