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
    save_losses()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def save_losses():
    np.save(output_dir+'/losses/'+checkpoint_str+"_train_losses",np.array(train_losses))
    np.save(output_dir+'/losses/'+checkpoint_str+"_valid_losses",np.array(valid_losses))
    plt.ylim([0,2])
    plt.plot(train_losses, label='training loss')
    plt.plot(valid_losses, label='validation loss')
    plt.xlabel('Training epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(output_dir+'/losses/'+checkpoint_str+'_losses.png')

def save_params():
    params={'Hidden Layers': hidden_layers,
            'Learning Rate': learning_rate,
            'Batch Size': batch_size,
            'Num Epochs': num_epochs}
    saveFile = output_dir + '/params.txt'
    with open(saveFile, 'w') as f:
        for key,value in params.items():
            f.write('%s:%s\n' % (key, value))


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
if cfg['model'] == 'VarioMLP':
    num_classes = cfg['num_classes']
    vario_num_lag = cfg['vario_num_lag']
    hidden_layers = cfg['hidden_layers']
    imSize = cfg['split_img_size']
    model = VarioMLP.VarioMLP(num_classes, vario_num_lag, hidden_layers=hidden_layers) 
    img_transforms_train = transforms.Compose([
        DirectionalVario(model.num_lag),
        RandomRotateVario(),
    ])
    img_transforms_valid = transforms.Compose([
        DirectionalVario(model.num_lag),
        DefaultRotateVario(),
    ])

elif cfg['model'] == 'Resnet18':
    num_classes = cfg['num_classes']
    image_folder = cfg['training_img_path']
    vario_num_lag = cfg['vario_num_lag']
    image_paths, variogram_data, labels = collect_image_paths_and_labels(image_folder)
    model = Resnet18.resnet18(pretrained=False, num_classes=num_classes)
    img_transforms_train = None
    img_transforms_valid = None

elif cfg['model'] == 'VarioNet': #Only works for training via images as of now
    num_classes = cfg['num_classes']
    image_folder = cfg['training_img_path']
    vario_num_lag = cfg['vario_num_lag']
    image_paths, variogram_data, labels = collect_image_paths_and_labels(image_folder)
    #resnet = Resnet18.resnet18(pretrained=False, num_classes=num_classes)

    # Calculate the size of the variogram data
    #variogram_size = variogram_data.shape[0] * variogram_data.shape[1] * vario_num_lag  # 212
    variogram_size = 1*4*vario_num_lag
    #model = CombinedModel(vario_num_lag, num_classes)
    
    img_transforms_train = None
    img_transforms_valid = None

elif cfg['model'] == 'DDAiceNet':
    ddaBool = True
    num_classes = cfg['num_classes']
    nres = cfg['nres']
    model = DDAiceNet.DDAiceNet(num_classes,nres*2, hiddenLayers=hidden_layers)
    img_transforms_train = None
    img_transforms_valid = None


else:
    print("Error: Model \'{}\' not recognized".format(cfg['model']))
    exit(1)
if cfg['model'] == 'VarioNet':
    print("VarioNet")
else:
    print(model)
if imgTrain:
    # Perform train/test split
    if cfg['model'] == 'VarioNet' or cfg['model'] == 'Resnet18':
        label_path = cfg['training_img_npy']
        labeled_data = np.load(label_path, allow_pickle=True)
        labelInfo = labeled_data[0]
        dataset_labeled = labeled_data[1]
        dataset = np.load(dataset_path, allow_pickle=True)
        dataset_info = dataset[0]
        dataset_coords = dataset[1]
        train_size = int(cfg['train_test_split'] * len(image_paths))
        train_indeces = np.random.choice(range(np.array(len(image_paths))), train_size, replace=False)
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
            train_coords.append(dataset_labeled[i])
        for i in test_indeces:
            test_imgs.append(image_paths[i])
            test_var.append(variogram_data[i])
            test_labels.append(labels[i])
            test_coords.append(dataset_labeled[[i]])
    else:
        label_path = cfg['training_img_npy']
        labeled_data = np.load(label_path, allow_pickle=True)
        labelInfo = labeled_data[0]
        dataset_labeled = labeled_data[1]
        dataset = np.load(dataset_path, allow_pickle=True)
        dataset_info = dataset[0]
        dataset_coords = dataset[1]

        train_size = int(cfg['train_test_split'] * dataset_labeled.shape[0])

        train_indeces = np.random.choice(range(np.array(dataset_labeled.shape[0])), train_size, replace=False)
        test_indeces = np.setdiff1d(range(np.array(dataset_labeled.shape[0])), train_indeces)
        #CST20240322 Creating loops so train and test coords aren't 1D
        train_coords = []
        test_coords = []

        for i in train_indeces:
            train_coords.append(dataset_labeled[i])
        for i in test_indeces:
            test_coords.append(dataset_labeled[[i]])
    
    print('----- Initializing Dataset -----')

    if cfg['model'] == 'VarioMLP':
        train_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = train_coords,
            train = True,
            transform = img_transforms_train
            )

        valid_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = test_coords,
            train = True,
            transform = img_transforms_valid
            )

    elif cfg['model'] == 'VarioNet':
        trainImages = load_images(train_imgs)
        testImages = load_images(test_imgs)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
        ])
        var_transforms_train = transforms.Compose([
        DirectionalVario(vario_num_lag),
        RandomRotateVario(),
        ])
        
        train_dataset = FromFolderDataset(cfg['model'], train_imgs, train_var, train_labels, transform)
        valid_dataset = FromFolderDataset(cfg['model'], test_imgs, test_var, test_labels, transform)

        vario_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = train_coords,
            train = True,
            transform = var_transforms_train
            )
        image_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = train_coords,
            train = True,
            transform = img_transforms_train
            )

    elif cfg['model'] == 'Resnet18':
        trainImages = load_images(train_imgs)
        testImages = load_images(test_imgs)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
        ])
        train_dataset = FromFolderDataset(cfg['model'], train_imgs, train_var, train_labels, transform)
        valid_dataset = FromFolderDataset(cfg['model'], test_imgs, test_var, test_labels, transform)
    else:
        train_dataset = DDAiceDataset(
            dataPath = topDir,
            dataInfo = dataset_info,
            dataLabeled = train_coords,
            train = True,
            transform = None
            )

        valid_dataset = DDAiceDataset(
            dataPath = topDir,
            dataInfo = dataset_info,
            dataLabeled = test_coords,
            train = True,
            transform = None
            )
         #CST20240315
    print('Training set size: \t%d images'%(len(train_dataset)))
    # for i in range(num_classes):
    #     print('Class {}: {} - {} train images'.format(i,classEnum[i],len(train_coords[train_coords[:,4] == i])))
    # print('Validation set size: \t%d images'%(len(valid_dataset)))
    # for i in range(num_classes):
    #     print('Class {}: {} - {} valid images'.format(i,classEnum[i],len(test_coords[test_coords[:,4] == i])))
    print('----- Initializing DataLoader -----')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
        )

    
    print("train loader", type(train_loader))
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False
        )

    weighted = False
    if weighted:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y2 = list(valid_dataset.get_labels())
            y1 = list(train_dataset.get_labels())
            y = y1 + y2
            print('Class 0: {}'.format(y.count(0.0)))
            print('Class 1: {}'.format(y.count(1.0)))
            print('Class 2: {}'.format(y.count(2.0)))
            # print('Class 3: {}'.format(y.count(3.0)))

            class_wts = compute_class_weight('balanced',np.unique(y),y)
            class_wts = torch.from_numpy(class_wts).float()
            criterion = torch.nn.CrossEntropyLoss(weight=class_wts)
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) #TODO: what this does?
    else:
        # Initialize loss critereron and gradient descent optimizer
        criterion = torch.nn.CrossEntropyLoss()
        if cfg['model'] != 'VarioNet':
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    # Load model checkpoint
    if args.load_checkpoint and cfg['model'] != 'VarioNet':
        checkpoint_path = args.load_checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    # Initialize cuda
    if args.cuda and cfg['model'] != 'VarioNet':
        print('----- Initializing CUDA -----')
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        model.cuda()
        #optimizer.cuda()

    # Create directory for model checkpoints and output
    print('----- Initializing Output Directory -----')
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y_%H:%M")
    config_str = args.config.split('/')[1]
    output_dir = 'Output/%s_%s'%(config_str, date_str)
    checkpoint_str = ''
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(output_dir+'/checkpoints'): os.mkdir(output_dir+'/checkpoints')
    if not os.path.exists(output_dir+'/labels'): os.mkdir(output_dir+'/labels')
    if not os.path.exists(output_dir+'/losses'): os.mkdir(output_dir+'/losses')
    print('Output saved at %s'%(output_dir))

    save_params()

    print('----- Training -----')

    train_losses = []
    valid_losses = []
    varioLoss = []
    # X = tensor
    # Y = label
    if cfg['model'] == 'VarioNet':
            sum_loss = 0
            print("Training VarioMLP")
            vario_mlp = VarioMLP.VarioMLP(num_classes, vario_num_lag, hidden_layers=hidden_layers) 
            optimizer_vario = torch.optim.Adam(vario_mlp.parameters(), lr=learning_rate)
            vario_loader = DataLoader(
                vario_dataset,
                batch_size=batch_size,
                shuffle=True
                )
            
            for epoch in range(num_epochs):
                vario_mlp.train()
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
                    

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            # Save VarioMLP model
            torch.save(vario_mlp.state_dict(), 'vario_mlp.pth')
            #Resnet
            print("Training Resnet18")
            resnet18 = Resnet18.resnet18(pretrained=False, num_classes=num_classes)
            optimizer_resnet = torch.optim.Adam(resnet18.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            image_loader = DataLoader(
                image_dataset,
                batch_size=batch_size,
                shuffle=True
                )
            # Train loop for ResNet18
            for epoch in range(num_epochs):
                resnet18.train()
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

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            # Save ResNet18 model
            torch.save(resnet18.state_dict(), 'resnet18.pth')



            vario_mlp.load_state_dict(torch.load('vario_mlp.pth'))
            resnet18.load_state_dict(torch.load('resnet18.pth'))
            combined_model = CombinedModel(vario_mlp, resnet18, num_classes)
            model = combined_model
            if args.cuda:
                print('----- Initializing CUDA -----')
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
                combined_model.cuda()
                #optimizer.cuda()
            print("Training VarioNet")
            optimizer = optim.Adam(combined_model.parameters(),lr=learning_rate)
            for epoch in range(num_epochs):
                combined_model.train()
                for images, variograms, labels in train_loader:
                    if args.cuda:
                        images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                    images = torch.unsqueeze(images,1).float()
                    variograms = torch.unsqueeze(variograms,1).float()
                    optimizer.zero_grad()
                    outputs = combined_model(images, variograms)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            for param in combined_model.parameters():
                param.requires_grad = True

            # Fine-tune with a smaller learning rate
            optimizer_finetune = torch.optim.Adam(combined_model.parameters(), lr=1e-6)

            # Fine-tuning loop
            print("Fine Tuning VarioNet")
            for epoch in range(fine_epochs):
                combined_model.train()
                for batch_idx, (images, variograms, labels) in enumerate(train_loader):
                    if args.cuda:
                        images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                    images = torch.unsqueeze(images,1).float()
                    variograms = torch.unsqueeze(variograms,1).float()
                    optimizer_finetune.zero_grad()
                    outputs = combined_model(images, variograms)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer_finetune.step()
                    sum_loss += loss.item()
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            train_losses.append(sum_loss/batch_idx)

            # Validation phase
            print('running validation')
            combined_model.eval()
            val_loss = 0.0
            correct = 0
            total_variogram_loss= 0.0
            varioLoss = []

            with torch.no_grad():
                for batch_idx, (images, variograms, labels) in enumerate(valid_loader):
                    if args.cuda:
                        images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                    images = torch.unsqueeze(images,1).float()
                    variograms = torch.unsqueeze(variograms,1).float()
                    
                    outputs = combined_model(images, variograms)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    #variogram_validation = model(images, variogram=variograms)
                    #variogram_loss = criterion(variogram_validation, labels)
                    #total_variogram_loss += variogram_loss.item()
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
       
            valid_losses.append(val_loss/batch_idx)
    for epoch in range(num_epochs):

        print("EPOCH: {} ".format(epoch),end='',flush=True)

        sum_loss = 0
        

        if cfg['model'] == 'VarioNet':
            print("")
        else:
            for batch_idx,(X,Y) in enumerate(train_loader):
                
                #CST20240315 make program exit and say train data set is too low
                if int((len(train_dataset) / batch_size)/10) != 0: #So it won't crash 
                    if batch_idx % int((len(train_dataset) / batch_size)/10) == 0:
                        print('.', end='',flush=True)
                else:
                    print("ERROR: The length of the training dataset is too small") #CST 20240318
                    sys.exit(0)
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
                Y_hat = model.forward(X)

                # Calculate training loss
                loss = criterion(Y_hat, Y)

                # Perform backprop and zero gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                sum_loss = sum_loss + float(criterion(Y_hat, Y))

                #print("EPOCH: %d\t BATCH: %d\tTRAIN LOSS = %f"%(epoch,batch_idx,loss.item()))
                #Make exit if batch_idx is zero
            if batch_idx != 0:
                train_losses.append(sum_loss/batch_idx)
            else:
                print("ERROR: The length of the training dataset is too small") #CST 20240318
                sys.exit(0)

            print('running validation')

            #Valid
            loss = 0
            for batch_idx,(X,Y) in enumerate(valid_loader):
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
                Y_hat = model.forward(X)

                # Calculate training loss
                loss = loss + float(criterion(Y_hat, Y))

            valid_losses.append(loss/batch_idx)
        
        
        print("\tTRAIN LOSS = {:.5f}\tVALID LOSS = {:.5f}".format(train_losses[-1],valid_losses[-1]))

        print('saving checkpoint')
        # Save checkpoint
        checkpoint_str = "epoch_" + str(epoch)
        if valid_losses[-1] == np.array(valid_losses).min():
            checkpoint_path = os.path.join(output_dir, 'checkpoints', checkpoint_str)
            checkpoint = {'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)
        else:
            if len(valid_losses) > np.array(valid_losses).argmin() + 100:
                break
    checkpoint_path = os.path.join(output_dir, 'checkpoints', checkpoint_str)
    checkpoint = {'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, checkpoint_path)
    save_losses()

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

    train_indeces = np.random.choice(range(np.array(dataset_labeled.shape[0])), train_size, replace=False)
    test_indeces = np.setdiff1d(range(np.array(dataset_labeled.shape[0])), train_indeces)
    #CST20240322 Creating loops so train and test coords aren't 1D
    train_coords = []
    test_coords = []
    for i in train_indeces:
        train_coords.append(dataset_labeled[i])
    for i in test_indeces:
        test_coords.append(dataset_labeled[[i]])
    #print("train size", train_size)#CST20240318
    #print("train_indeces", train_indeces) #CST20240315
    #print("dataset labeled", dataset_labeled) #CST20240315
    #print("train coords", train_coords) #CST20240315

    # Initialize Datasets and DataLoaders
    print('----- Initializing Dataset -----')

    if cfg['model'] == 'VarioMLP' or cfg['model'] == 'Resnet18':
        train_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = train_coords,
            train = True,
            transform = img_transforms_train
            )

        valid_dataset = SplitImageDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = test_coords,
            train = True,
            transform = img_transforms_valid
            )
    elif cfg['model'] == 'VarioNet':
        train_dataset = TestDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = train_coords,
            train = True,
        )

        valid_dataset = TestDataset(
            imgPath = topDir,
            imgData = dataset_info,
            labels = test_coords,
            train = True,
            )
    else:
        train_dataset = DDAiceDataset(
            dataPath = topDir,
            dataInfo = dataset_info,
            dataLabeled = train_coords,
            train = True,
            transform = None
            )

        valid_dataset = DDAiceDataset(
            dataPath = topDir,
            dataInfo = dataset_info,
            dataLabeled = test_coords,
            train = True,
            transform = None
            )

    #CST20240315
    print('Training set size: \t%d images'%(len(train_dataset)))
    # for i in range(num_classes):
    #     print('Class {}: {} - {} train images'.format(i,classEnum[i],len(train_coords[train_coords[:,4] == i])))
    # print('Validation set size: \t%d images'%(len(valid_dataset)))
    # for i in range(num_classes):
    #     print('Class {}: {} - {} valid images'.format(i,classEnum[i],len(test_coords[test_coords[:,4] == i])))
    print('----- Initializing DataLoader -----')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
        )
    print("train loader", type(train_loader))
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False
        )

    weighted = False
    if weighted:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y2 = list(valid_dataset.get_labels())
            y1 = list(train_dataset.get_labels())
            y = y1 + y2
            print('Class 0: {}'.format(y.count(0.0)))
            print('Class 1: {}'.format(y.count(1.0)))
            print('Class 2: {}'.format(y.count(2.0)))
            # print('Class 3: {}'.format(y.count(3.0)))

            class_wts = compute_class_weight('balanced',np.unique(y),y)
            class_wts = torch.from_numpy(class_wts).float()
            criterion = torch.nn.CrossEntropyLoss(weight=class_wts)
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) #TODO: what this does?
    else:
        # Initialize loss critereron and gradient descent optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    # Load model checkpoint
    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    # Initialize cuda
    if args.cuda:
        print('----- Initializing CUDA -----')
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        model.cuda()
        #optimizer.cuda()

    # Create directory for model checkpoints and output
    print('----- Initializing Output Directory -----')
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y_%H:%M")
    config_str = args.config.split('/')[1]
    output_dir = 'Output/%s_%s'%(config_str, date_str)
    checkpoint_str = ''
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(output_dir+'/checkpoints'): os.mkdir(output_dir+'/checkpoints')
    if not os.path.exists(output_dir+'/labels'): os.mkdir(output_dir+'/labels')
    if not os.path.exists(output_dir+'/losses'): os.mkdir(output_dir+'/losses')
    print('Output saved at %s'%(output_dir))

    save_params()

    print('----- Training -----')

    train_losses = []
    valid_losses = []

    # X = tensor
    # Y = label

    for epoch in range(num_epochs):

        print("EPOCH: {} ".format(epoch),end='',flush=True)

        sum_loss = 0
        if cfg['model'] == 'VarioNet':
            # Training phase
            for batch_idx, (images, variograms, labels) in enumerate(train_loader):
                if int((len(train_dataset) / batch_size)/10) != 0: #So it won't crash 
                    if batch_idx % int((len(train_dataset) / batch_size)/10) == 0:
                        print('.', end='',flush=True)
                else:
                    print("ERROR: The length of the training dataset is too small") #CST 20240318
                    sys.exit(0)
                if args.cuda:
                    images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                    images = torch.unsqueeze(images,1).float()
                    #images = images.view(images.size(0), images.size(2), images.size(3), images.size(4))
                    variograms = torch.unsqueeze(variograms,1).float()
                else:
                    images = torch.unsqueeze(images,1).float()
                    #images = images.view(images.size(0), images.size(2), images.size(3), images.size(4))
                    variograms = torch.unsqueeze(variograms,1).float()
               
                optimizer.zero_grad()
                outputs = model(images, variograms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
            train_losses.append(sum_loss/batch_idx)
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0

            with torch.no_grad():
                for batch_idx, (images, variograms, labels) in enumerate(valid_loader):
                    if args.cuda:
                        images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                        images = torch.unsqueeze(images,1).float()
                        #images = images.view(images.size(0), images.size(2), images.size(3), images.size(4))
                        variograms = torch.unsqueeze(variograms,1).float()
                    else:
                        images = torch.unsqueeze(images,1).float()
                        #images = images.view(images.size(0), images.size(2), images.size(3), images.size(4))
                        variograms = torch.unsqueeze(variograms,1).float()
                    outputs = model(images, variograms)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
       
            valid_losses.append(val_loss/batch_idx)
        else:
            for batch_idx,(X,Y) in enumerate(train_loader):
                
                #CST20240315 else make program exit and say train data set is too low
                if int((len(train_dataset) / batch_size)/10) != 0: #So it won't crash 
                    if batch_idx % int((len(train_dataset) / batch_size)/10) == 0:
                        print('.', end='',flush=True)
                else:
                    print("ERROR: The length of the training dataset is too small") #CST 20240318
                    sys.exit(0)

                # Move batch to GPU
                if args.cuda:
                    X,Y = X.to(device),Y.to(device)
                    #X = X.view((X.shape[0],1,-1)).float()
                    X = torch.unsqueeze(X,1).float()
                else:
                    #X = X.view((X.shape[0],1,-1)).float()
                    X = torch.unsqueeze(X,1).float()

                # Compute forward pass
                Y_hat = model.forward(X)

                # Calculate training loss
                loss = criterion(Y_hat, Y)

                # Perform backprop and zero gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                sum_loss = sum_loss + float(criterion(Y_hat, Y))

                #print("EPOCH: %d\t BATCH: %d\tTRAIN LOSS = %f"%(epoch,batch_idx,loss.item()))
                #Make exit if batch_idx is zero
            if batch_idx != 0:
                train_losses.append(sum_loss/batch_idx)
            else:
                print("ERROR: The length of the training dataset is too small") #CST 20240318
                sys.exit(0)


            print('running validation')

            #Valid
            loss = 0
            for batch_idx,(X,Y) in enumerate(valid_loader):
                # Move batch to GPU
                if args.cuda:
                    X,Y = X.to(device),Y.to(device)
                    #X = X.view((X.shape[0],1,-1)).float()
                    X = torch.unsqueeze(X,1).float()
                else:
                    #X = X.view((X.shape[0],1,-1)).float()
                    X = torch.unsqueeze(X,1).float()

                # Compute forward pass
                Y_hat = model.forward(X)

                # Calculate training loss
                loss = loss + float(criterion(Y_hat, Y))

        valid_losses.append(loss/batch_idx)
        print("\tTRAIN LOSS = {:.5f}\tVALID LOSS = {:.5f}".format(train_losses[-1],valid_losses[-1]))

        print('saving checkpoint')
        # Save checkpoint
        checkpoint_str = "epoch_" + str(epoch)
        if valid_losses[-1] == np.array(valid_losses).min():
            checkpoint_path = os.path.join(output_dir, 'checkpoints', checkpoint_str)
            checkpoint = {'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)
        else:
            if len(valid_losses) > np.array(valid_losses).argmin() + 100:
                break
    checkpoint_path = os.path.join(output_dir, 'checkpoints', checkpoint_str)
    checkpoint = {'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, checkpoint_path)
    save_losses()
















    """
            # Training phase
            for batch_idx, (images, variograms, labels) in enumerate(train_loader):
                if int((len(train_dataset) / batch_size)/10) != 0: #So it won't crash 
                    if batch_idx % int((len(train_dataset) / batch_size)/10) == 0:
                        print('.', end='',flush=True)
                else:
                    print("ERROR: The length of the training dataset is too small") #CST 20240318
                    sys.exit(0)
                if args.cuda:
                    images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                    images = torch.unsqueeze(images,1).float()
                    #images = images.view(images.size(0), images.size(2), images.size(3), images.size(4))
                    variograms = torch.unsqueeze(variograms,1).float()
                else:
                    images = torch.unsqueeze(images,1).float()
                    #images = images.view(images.size(0), images.size(2), images.size(3), images.size(4))
                    variograms = torch.unsqueeze(variograms,1).float()
               
                optimizer.zero_grad()
                outputs = model(images, variograms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            train_losses.append(sum_loss/batch_idx)
            # Validation phase
            print('running validation')
            model.eval()
            val_loss = 0.0
            correct = 0
            total_variogram_loss= 0.0
            varioLoss = []

            with torch.no_grad():
                for batch_idx, (images, variograms, labels) in enumerate(valid_loader):
                    if args.cuda:
                        images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                        images = torch.unsqueeze(images,1).float()
                        #images = images.view(images.size(0), images.size(2), images.size(3), images.size(4))
                        variograms = torch.unsqueeze(variograms,1).float()
                    else:
                        images = torch.unsqueeze(images,1).float()
                        #images = images.view(images.size(0), images.size(2), images.size(3), images.size(4))
                        variograms = torch.unsqueeze(variograms,1).float()
                    outputs = model(images, variograms)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    #variogram_validation = model(images, variogram=variograms)
                    #variogram_loss = criterion(variogram_validation, labels)
                    #total_variogram_loss += variogram_loss.item()
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
       
            valid_losses.append(val_loss/batch_idx)
            #varioLoss.append(total_variogram_loss/batch_idx)
"""