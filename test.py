from torch.utils.data import DataLoader
import torch
from Dataset import *
from torch import optim
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime
import random
from Models import *
import yaml

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--netCDF", action="store_true")
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Set training hyperparameters as specified by config file
learning_rate = float(cfg['learning_rate'])
batch_size = cfg['batch_size']
num_epochs = cfg['num_epochs']

# Set dataset hyperparameters as specified by config file
topDir = cfg['img_path']
dataset_path = cfg['npy_path']
train_path = cfg['train_path']
valid_path = cfg['valid_path']

# Initialize NN model as specified by config file
print('----- Initializing Neural Network Model -----')

if cfg['model'] == 'VarioMLP':
    num_classes = cfg['num_classes']
    vario_num_lag = cfg['vario_num_lag']
    hidden_layers = cfg['hidden_layers']
    model = VarioMLP.VarioMLP(num_classes, vario_num_lag, hidden_layers=hidden_layers)
    img_transforms_valid = transforms.Compose([
        DirectionalVario(model.num_lag),
        DefaultRotateVario(),
    ])
elif cfg['model'] == 'Resnet18':
    num_classes = cfg['num_classes']
    model = Resnet18.resnet18(pretrained=False, num_classes=num_classes)
    img_transforms_valid = None

elif cfg['model'] == 'DDAiceNet':
    ddaBool = True
    num_classes = cfg['num_classes']
    nres = cfg['nres']
    model = DDAiceNet.DDAiceNet(num_classes,nres-1)
    img_transforms_valid = None
else:
    print("Error: Model \'%s\' not recognized"%(cfg['model']))
    exit(1)

print(model)

# Load model checkpoint
if args.load_checkpoint:
    checkpoint_path = args.load_checkpoint
    split = checkpoint_path.split('/')
    checkpoint_str = split[-1]
    output_dir = split[0]+'/'+split[1]
    print(checkpoint_str, output_dir)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

else:
    print("Please specify a model checkpoint with the --load_checkpoint argument")
    exit(1)

# Initialize Datasets and DataLoaders
print('----- Initializing Dataset -----')
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
dataset_labels = dataset[1]

if cfg['model'] == 'VarioMLP' or cfg['model'] == 'Resnet18':
    valid_dataset = SplitImageDataset(
        imgPath = topDir,
        imgData = dataset_info,
        labels = dataset_labels,
        train = False,
        transform = img_transforms_valid
        )
else:
    valid_dataset = DDAiceDataset(
        dataPath = topDir,
        dataInfo = dataset_info,
        dataLabeled = dataset_labels,
        cutoff = nres-1,
        train = False,
        transform = None
        )


print('Test set size: \t%d images'%(len(valid_dataset)))

print('----- Initializing DataLoader -----')

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False
)

# Initialize loss critereron and gradient descent optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

# Initialize cuda
if args.cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model.cuda()
    #optimizer.cuda()

# Constants from Authors100 dataset
MAXWIDTH = 2260
MAXHEIGHT = 337

softmax = torch.nn.Softmax(dim=1)

print('----- Initializing DataLoader -----')


print('Done!')

print('----- Training -----')

labels = []
confs = []

for batch_idx,X in enumerate(valid_loader):

    if batch_idx % 100 == 0:
        print(batch_idx)

    # Move batch to GPU
    if args.cuda:
        X = X.to(device)

    X = torch.unsqueeze(X,1).float()

    # Compute forward pass
    Y_hat = model.forward(X)

    sm = softmax(Y_hat)

    conf = sm.max()

    if conf > 0:
        labels.append(torch.argmax(Y_hat))
        confs.append(conf.item())
    else:
        labels.append(num_classes)

#dataset[0]['filename'] = topDir

split_info = dataset[1]
if ddaBool:
    split_info[:,nres-1] = labels
    split_info[:,nres] = confs
else:
    split_info[:,4] = labels
    split_info[:,5] = confs
#split_info = np.concatenate((split_info, np.array(confs).reshape(len(confs),1)),1)

print(dataset[1].shape)
dataset[1] = split_info
#data.append(confs)
np.save(output_dir+"/labels/labeled_"+checkpoint_str, dataset)

if args.netCDF:
    to_netCDF(dataset)
