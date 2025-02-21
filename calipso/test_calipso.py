import sys
sys.path.append('/home/twickler/ws/GEOCLASS-image/NN_Class')
from torch.utils.data import DataLoader
import torch
from Dataset import *
from torch import optim
from torchvision import transforms
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime
import random
from Models import VarioMLP
from Models import CalipsoMLP
from VarioNet import CombinedModel
from VarioNet import collect_image_paths_and_labels
from VarioNet import load_images
from VarioNet import TestDataset
import yaml
import warnings
from torchvision import models
import rasterio as rio
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

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
alpha = cfg['alpha']
beta = cfg['beta']

# Initialize NN model as specified by config file
print('----- Initializing Neural Network Model -----')
ddaBool = False
if cfg['model'] == 'CalipsoMLP':
    num_classes = cfg['num_classes']
    hidden_layers = cfg['hidden_layers']
    channels = cfg['num_channels']
    imSize = cfg['split_img_size']
    image_folder = cfg['training_img_path']
    density_size = imSize[0]*imSize[1]
    model = CalipsoMLP.CalipsoMLP(num_classes, channels, density_size, hidden_layers=hidden_layers) 
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

winSize = cfg['split_img_size']
density = []
tab = []
asr = []
density_path = cfg['density_path']
density_data = np.loadtxt(density_path, dtype= float)
density = density_data
tab_paths = cfg['tab_path']
for tab_path in tab_paths:
    tab_data = np.loadtxt(tab_path, dtype= float)
    tab.append(tab_data)
asr_paths = cfg['asr_path']
for asr_path in asr_paths:
    asr_data = np.loadtxt(asr_path, dtype= float)
    asr.append(asr_data)


valid_dataset = CalipsoDataset2(
    imgPath = topDir,
    imgData = dataset_info,
    labels = dataset_labels,
    den = density,
    tab = tab,
    asr =asr,
    tile_width = winSize[0],
    tile_height = winSize[1],
    train = False,
    transform = None
    )

print('\nTest set size: \t%d images'%(len(valid_dataset)))

print('----- Initializing DataLoader -----')

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False
)

weighted = False
if weighted:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y = list(valid_dataset.get_labels())
        print('Class 0: {}'.format(y.count(0.0)))
        print('Class 1: {}'.format(y.count(1.0)))
        print('Class 2: {}'.format(y.count(2.0)))
        print('Class 3: {}'.format(y.count(3.0)))

        class_wts = compute_class_weight('balanced',np.unique(y),y)
        class_wts = torch.from_numpy(class_wts).float()
        criterion = torch.nn.CrossEntropyLoss(weight=class_wts)
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
else:
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
        print(f"Processing batch {batch_idx}")
            
    # Move batch to GPU
    if args.cuda:
        X = X.to(device)
    #X = torch.unsqueeze(X,1).float()

    # Compute forward pass
    Y_hat = model.forward(X)

    sm = softmax(Y_hat)

    conf = sm.max()

    if conf > 0:
        labels.append(int(torch.argmax(Y_hat)))
        confs.append(conf.item())
    else:
        labels.append(num_classes)

#dataset[0]['filename'] = topDir

split_info = dataset[1]
if ddaBool:
    split_info[:,0] = labels
    split_info[:,1] = confs
else:
    split_info[:,2] = labels
    split_info[:,3] = confs
#split_info = np.concatenate((split_info, np.array(confs).reshape(len(confs),1)),1)

print(dataset[1].shape)
dataset[1] = split_info
#data.append(confs)
np.save(output_dir+"/labels/labeled_"+checkpoint_str, dataset)

if args.netCDF:
    to_netCDF(dataset)