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
from VarioNet import *
import yaml
import warnings
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
hidden_layers = cfg['hidden_layers']
imgTrain = cfg['train_with_img']
adapt = cfg['adaptive']

# Set dataset hyperparameters as specified by config file
topDir = cfg['img_path']
dataset_path = cfg['npy_path']
train_path = cfg['train_path']
valid_path = cfg['valid_path']

# Initialize NN model as specified by config file
print('----- Initializing Neural Network Model -----')
ddaBool = False
if cfg['model'] == 'VarioMLP':
    num_classes = cfg['num_classes']
    vario_num_lag = cfg['vario_num_lag']
    hidden_layers = cfg['hidden_layers']
    imSize = cfg['split_img_size']
    model = VarioMLP.VarioMLP(num_classes, vario_num_lag, hidden_layers=hidden_layers) 
    img_transforms_valid = transforms.Compose([
        DirectionalVario(model.num_lag),
        DefaultRotateVario(),
    ])
elif cfg['model'] == 'Resnet18':
    num_classes = cfg['num_classes']
    model = Resnet18.resnet18(pretrained=False, num_classes=num_classes)
    img_transforms_valid = None
elif cfg['model'] == 'VarioNet':
    num_classes = cfg['num_classes']
    vario_num_lag = cfg['vario_num_lag']
    image_folder = cfg['training_img_path']
    alpha = cfg['alpha']
    beta = cfg['beta']
    vario_mlp = VarioMLP.VarioMLP(num_classes, vario_num_lag, hidden_layers=hidden_layers)
    resnet18 = Resnet18.resnet18(pretrained=False, num_classes=num_classes)
    vario_mlp.load_state_dict(torch.load('vario_mlp.pth'))
    resnet18.load_state_dict(torch.load('resnet18.pth'))
    model = CombinedModel(vario_mlp, resnet18, num_classes, a = alpha, b = beta, adaptive=adapt)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
        ])

elif cfg['model'] == 'DDAiceNet':
    ddaBool = True
    num_classes = cfg['num_classes']
    nres = cfg['nres']
    hidden_layers = cfg['hidden_layers']
    model = DDAiceNet.DDAiceNet(num_classes, nres*2, hiddenLayers=hidden_layers)
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
elif cfg['model'] == 'VarioNet':
    
    valid_dataset = TestDataset(
        imgPath = topDir,
        imgData = dataset_info,
        labels = dataset_labels,
        train = False,
        transform = transforms.Compose([
        DirectionalVario(vario_num_lag),
        DefaultRotateVario(),
    ])
        )
else:
    valid_dataset = DDAiceDataset(
        dataPath = topDir,
        dataInfo = dataset_info,
        dataLabeled = dataset_labels,
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


if cfg['model'] == 'VarioNet':

    for batch_idx, (images, variograms) in enumerate(valid_loader):
            
        if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}")

            # Move data to GPU
        if args.cuda:
                images = images.to(device)
                variograms = variograms.to(device)

            # Unsqueeze if needed (add a channel dimension for grayscale images)
        images = torch.unsqueeze(images, 1).float()
        variograms = variograms.float()  # Ensure variograms are floats

            # Compute forward pass through the combined model
        Y_hat = model.forward(images, variograms)
            # Apply softmax to get probabilities
        sm = softmax(Y_hat)

            # Get the max confidence score and corresponding label
        conf = sm.max()

        if conf > 0:
                labels.append(int(torch.argmax(Y_hat)))
                confs.append(conf.item())
        else:
                labels.append(num_classes)
else:              
    for batch_idx,X in enumerate(valid_loader):

        if batch_idx % 100 == 0:
            print(f"Processing batch {batch_idx}")
                
        # Move batch to GPU
        if args.cuda:
            X = X.to(device)

        X = torch.unsqueeze(X,1).float()

        # Compute forward pass
        Y_hat = model.forward(X)

        sm = softmax(Y_hat)

        conf = sm.max()

        if conf > 0:
            labels.append(int(torch.argmax(Y_hat)))
            confs.append(conf.item())
        else:
            labels.append(num_classes)

split_info = dataset[1]
if ddaBool:
    split_info[:,0] = labels
    split_info[:,1] = confs
else:
    split_info[:,4] = labels
    split_info[:,5] = confs
#split_info = np.concatenate((split_info, np.array(confs).reshape(len(confs),1)),1)

print(dataset[1].shape)
dataset[1] = split_info
#data.append(confs)
np.save(output_dir+"/labels/labeled_"+checkpoint_str, dataset)

if adapt and cfg['model'] == 'VarioNet':
     model.plot_beta(output_dir=output_dir, conf = confs)

if args.netCDF:
    to_netCDF(dataset)