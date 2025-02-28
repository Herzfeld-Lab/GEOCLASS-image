from torch.utils.data import DataLoader
from Dataset import *
from torch import optim
from torchvision import transforms
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime
import random
from Models import *
from VarioNet import CombinedModel
from VarioNet import collect_image_paths_and_labels
from VarioNet import load_images
from VarioNet import FromFolderDataset
from VarioNet import *
import yaml
import signal
from sklearn.utils.class_weight import compute_class_weight
import warnings
from sklearn.utils.class_weight import compute_class_weight
import warnings
from torchvision import models
import rasterio as rio
from PIL import Image
from VarioNet import TestDataset
import torch.nn.functional as F

    

# Handle Ctrl-C event (manual stop training)
def signal_handler(sig, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("--load_checkpoint", type=str, default=None)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Set training hyperparameters as specified by config file
learning_rate = float(cfg['learning_rate'])
batch_size = cfg['batch_size']
num_epochs = cfg['num_epochs']
fine_epochs = cfg['fine_epochs']
hidden_layers = cfg['hidden_layers']
imgTrain = cfg['train_with_img']

# Set dataset hyperparameters as specified by config file
topDir = cfg['img_path']
classEnum = cfg['class_enum']
dataset_path = cfg['npy_path']
train_path = cfg['train_path']
valid_path = cfg['valid_path']

# Initialize NN model as specified by config file
print('----- Initializing Neural Network Model -----')
#initializing ddaBool
ddaBool = False
num_classes = cfg['num_classes']
image_folder = cfg['training_img_path']
vario_num_lag = cfg['vario_num_lag']

    #resnet = Resnet18.resnet18(pretrained=False, num_classes=num_classes)

    # Calculate the size of the variogram data
    #variogram_size = variogram_data.shape[0] * variogram_data.shape[1] * vario_num_lag  # 212
    #model = CombinedModel(vario_num_lag, num_classes)
    
img_transforms_train = None
img_transforms_valid = None





if imgTrain:
    image_paths, variogram_data, labels = collect_image_paths_and_labels(image_folder)

    train_size = int(cfg['train_test_split'] * len(image_paths))
    if cfg['train_indices'] == 'None':
        train_indeces = np.random.choice(range(np.array(len(image_paths))), train_size, replace=False)
        z=len(train_indeces)
        dataset_path = args.config[:-7] + "_%d"%(num_classes)+"_%d"%(z)+"train_indeces"
        np.save(dataset_path, train_indeces)
        cfg['train_indices'] = dataset_path+'.npy'
        f = open(args.config, 'w')
        f.write(generate_config_silas(cfg))
        f.close()
    else:
         train_indeces_npy = cfg['train_indeces']
         train_indeces = np.load(train_indeces_npy)
    test_indeces = np.setdiff1d(range(np.array(len(image_paths))), train_indeces)
    #CST20240322 Creating loops so train and test coords aren't 1D
    train_imgs = []
    test_imgs = []
    train_var = []
    test_var = []
    train_labels = []
    test_labels = []
    train_coords = []
    test_coords = []

    for i in train_indeces:
        train_imgs.append(image_paths[i])
        train_var.append(variogram_data[i])
        train_labels.append(labels[i])
        #train_coords.append(dataset_labeled[i])
    for i in test_indeces:
        test_imgs.append(image_paths[i])
        test_var.append(variogram_data[i])
        test_labels.append(labels[i])
        #test_coords.append(dataset_labeled[[i]])
    
    print('----- Initializing Dataset -----')


    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
        ])
    var_transforms = transforms.Compose([
        DirectionalVario(vario_num_lag),
        RandomRotateVario(),
        ])
        


    vario_dataset = FromFolderDataset('VarioMLP', train_imgs, train_var, train_labels, None)
    vario_valid_dataset = FromFolderDataset('VarioMLP', train_imgs, train_var, train_labels, None)
    image_dataset = FromFolderDataset('Resnet18', train_imgs, train_var, train_labels, transform)
    image_valid_dataset = FromFolderDataset('Resnet18', test_imgs, test_var, test_labels, transform)
         #CST20240315
else:
    # Perform train/test split
    dataset = np.load(dataset_path, allow_pickle=True)
    dataset_info = dataset[0]
    dataset_coords = dataset[1]
    if ddaBool:
        dataset_labeled = dataset_coords[dataset_coords[:,0] != -1]
    else:
        dataset_labeled = dataset_coords[dataset_coords[:,4] != -1]

    train_size = int(cfg['train_test_split'] * dataset_labeled.shape[0])
    if cfg['train_indices'] == 'None':
        train_indeces = np.random.choice(range(np.array(dataset_labeled.shape[0])), train_size, replace=False)
        z=len(train_indeces)
        dataset_path = args.config[:-7] + "_%d"%(num_classes)+"_%d"%(z)+"train_indeces"
        np.save(dataset_path, train_indeces)
        cfg['train_indeces'] = dataset_path+'.npy'
        f = open(args.config, 'w')
        f.write(generate_config_silas(cfg))
        f.close()
    else:
         train_indeces_npy = cfg['train_indices']
         train_indeces = np.load(train_indeces_npy)
    test_indeces = np.setdiff1d(range(np.array(dataset_labeled.shape[0])), train_indeces)
    #CST20240322 Creating loops so train and test coords aren't 1D
    train_coords = []
    test_coords = []
    for i in train_indeces:
        train_coords.append(dataset_labeled[i])
    for i in test_indeces:
        test_coords.append(dataset_labeled[[i]])
    
    print('----- Initializing Dataset -----')


    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
        ])
    var_transforms = transforms.Compose([
        DirectionalVario(vario_num_lag),
        RandomRotateVario(),
        ])
    
    vario_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = train_coords,
            train = True,
            transform = var_transforms
            )
    vario_valid_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = test_coords,
            train = True,
            transform = var_transforms
            )
    image_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = train_coords,
            train = True,
            transform = None
            )
    image_valid_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = train_coords,
            train = True,
            transform = None
            )
print('Training set size: \t%d images'%(len(image_dataset)))
    # for i in range(num_classes):
    #     print('Class {}: {} - {} train images'.format(i,classEnum[i],len(train_coords[train_coords[:,4] == i])))
    # print('Validation set size: \t%d images'%(len(valid_dataset)))
    # for i in range(num_classes):
    #     print('Class {}: {} - {} valid images'.format(i,classEnum[i],len(test_coords[test_coords[:,4] == i])))
print('----- Initializing DataLoader -----')
    
criterion = torch.nn.CrossEntropyLoss()
    

print('----- Training -----')

train_losses = []
valid_losses = []
    # X = tensor
    # Y = label

    
print("Training VarioMLP")
vario_mlp = VarioMLP.VarioMLP(num_classes, vario_num_lag, hidden_layers=hidden_layers) 
optimizer_vario = torch.optim.Adam(vario_mlp.parameters(), lr=learning_rate)
if args.cuda:
        print('----- Initializing CUDA -----')
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        vario_mlp.cuda()
        #optimizer.cuda()
vario_loader = DataLoader(
                vario_dataset,
                batch_size=batch_size,
                shuffle=True
                )
vario_valid_loader =  DataLoader(
                vario_valid_dataset,
                batch_size=batch_size,
                shuffle=False
                )
for epoch in range(num_epochs):
        sum_loss = 0
        print("EPOCH: {} ".format(epoch),end='',flush=True)
        for batch_idx, (X,Y) in enumerate(vario_loader):
            if args.cuda:
                X,Y = X.to(device),Y.to(device)
            X = torch.unsqueeze(X,1).float()
            Y_hat = vario_mlp.forward(X)
            # Calculate training loss
            loss = criterion(Y_hat, Y)
            # Perform backprop and zero gradient
            optimizer_vario.zero_grad()
            loss.backward()
            optimizer_vario.step()
            optimizer_vario.zero_grad()
            sum_loss = sum_loss + float(criterion(Y_hat, Y))
        train_losses.append(sum_loss/batch_idx)
        print('running validation')
        #Valid
        loss = 0
        for batch_idx,(X,Y) in enumerate(vario_valid_loader):
            # Move batch to GPU
            if args.cuda:
                X,Y = X.to(device),Y.to(device)
                #X = X.view((X.shape[0],1,-1)).float()
                X = torch.unsqueeze(X,1).float()
            else:
                #X = X.view((X.shape[0],1,-1)).float()
                X = torch.unsqueeze(X,1).float()
            #if cfg['model'] == 'Resnet18':
                    #X = X.view(X.size(0), X.size(2), X.size(3), X.size(4))
                # Compute forward pass
            Y_hat = vario_mlp.forward(X)

                # Calculate training loss
            loss = loss + float(criterion(Y_hat, Y))

        valid_losses.append(loss/batch_idx)
        
        
        print("\tTRAIN LOSS = {:.5f}\tVALID LOSS = {:.5f}".format(train_losses[-1],valid_losses[-1]))

    # Save VarioMLP model
print("Saving VarioMLP to vario_mlp.pth")
torch.save(vario_mlp.state_dict(), 'vario_mlp.pth')
#Resnet
train_losses = []
valid_losses = []
print("Training Resnet18")
resnet18 = Resnet18.resnet18(pretrained=False, num_classes=num_classes)
optimizer_resnet = torch.optim.Adam(resnet18.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
if args.cuda:
        print('----- Initializing CUDA -----')
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        resnet18.cuda()
        #optimizer.cuda()
image_loader = DataLoader(
                image_dataset,
                batch_size=batch_size,
                shuffle=True
                )
image_valid_loader =  DataLoader(
                image_valid_dataset,
                batch_size=batch_size,
                shuffle=False
                )
    # Train loop for ResNet18
for epoch in range(num_epochs):
        sum_loss = 0
        print("EPOCH: {} ".format(epoch),end='',flush=True)
        for batch_idx, (X,Y) in enumerate(image_loader):
            if args.cuda:
                X,Y = X.to(device),Y.to(device)
            X = torch.unsqueeze(X,1).float()
            Y_hat = resnet18.forward(X)
            # Calculate training loss
            loss = criterion(Y_hat, Y)
            # Perform backprop and zero gradient
            optimizer_resnet.zero_grad()
            loss.backward()
            optimizer_resnet.step()
            optimizer_resnet.zero_grad()
            sum_loss = sum_loss + float(criterion(Y_hat, Y))
        train_losses.append(sum_loss/batch_idx)
        print('running validation')
            #Valid
        loss = 0
        for batch_idx,(X,Y) in enumerate(image_valid_loader):
            # Move batch to GPU
            if args.cuda:
                X,Y = X.to(device),Y.to(device)
                    #X = X.view((X.shape[0],1,-1)).float()
                X = torch.unsqueeze(X,1).float()
            else:
                #X = X.view((X.shape[0],1,-1)).float()
                X = torch.unsqueeze(X,1).float()
                #if cfg['model'] == 'Resnet18':
                    #X = X.view(X.size(0), X.size(2), X.size(3), X.size(4))
                # Compute forward pass
            Y_hat = resnet18.forward(X)

             # Calculate training loss
            loss = loss + float(criterion(Y_hat, Y))

        valid_losses.append(loss/batch_idx)
        
        
        print("\tTRAIN LOSS = {:.5f}\tVALID LOSS = {:.5f}".format(train_losses[-1],valid_losses[-1]))

# Save ResNet18 model
print("Saving Resnet18 to resnet18.pth")
torch.save(resnet18.state_dict(), 'resnet18.pth')
