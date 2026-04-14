from torch.utils.data import DataLoader
from Dataset_MS import *
from torch import optim
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from utils_MS import *
from datetime import datetime
import random
from VarioNet_MS import *
from Models import *
from Models.patchMLP import PatchMLP
import yaml
import signal
from sklearn.utils.class_weight import compute_class_weight
import warnings
from sklearn.utils.class_weight import compute_class_weight
import warnings



# Handle Ctrl-C event (manual stop training)
def signal_handler(sig, frame):
    save_losses("Training_Canceled")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def save_losses(checkpoint_str):
    losses_dir = os.path.join(output_dir, "losses")
    np.save(os.path.join(losses_dir, f"{checkpoint_str}_train_losses"), np.array(train_losses))
    np.save(os.path.join(losses_dir, f"{checkpoint_str}_valid_losses"), np.array(valid_losses))
    plt.ylim([0,2])
    plt.plot(train_losses, label='training loss')
    plt.plot(valid_losses, label='validation loss')
    plt.xlabel('Training epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(losses_dir, f"{checkpoint_str}_losses.png"))

def save_params():
    params={'Hidden Layers': hidden_layers,
            'Learning Rate': learning_rate,
            'Batch Size': batch_size,
            'Num Epochs': num_epochs}
    saveFile = os.path.join(output_dir, "params.txt")
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
set_cfg(cfg)

# Set training hyperparameters as specified by config file
learning_rate = float(cfg['learning_rate'])
batch_size = cfg['batch_size']
num_epochs = cfg['num_epochs']
hidden_layers = cfg['hidden_layers']
imgTrain = cfg['train_with_img']
fine_epochs = cfg['fine_epochs']
adapt = cfg['adaptive']

# Set dataset hyperparameters as specified by config file
topDir = cfg['img_path']
classEnum = cfg.get('class_enum', cfg.get('class_enum'))
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
    image_folder = cfg['training_img_path']
    model = VarioMLP.VarioMLP(num_classes, vario_num_lag, hidden_layers=hidden_layers) 
    img_transforms_train = transforms.Compose([
        DirectionalVario(model.num_lag),
        RandomRotateVario(),
    ])
    img_transforms_valid = transforms.Compose([
        DirectionalVario(model.num_lag),
        DefaultRotateVario(),
    ])

elif cfg['model'] == 'wri_MLP':
    num_classes = cfg['num_classes']
    vario_num_lag = cfg['vario_num_lag']
    hidden_layers = cfg['hidden_layers']
    imSize = cfg['split_img_size']
    image_folder = cfg['training_img_path']
    activation = cfg['activation']
    #Dropout is currently hardcoded to be 0 within PatchMLP
    img_transforms_train = None
    img_transforms_valid = None

elif cfg['model'] == 'Resnet18':
    num_classes = cfg['num_classes']
    image_folder = cfg['training_img_path']
    model = Resnet18.resnet18(pretrained=False, num_classes=num_classes)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
        ])
    img_transforms_train = None
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
    model = DDAiceNet.DDAiceNet(num_classes,nres*2, hiddenLayers=hidden_layers)
    img_transforms_train = None
    img_transforms_valid = None


else:
    print("Error: Model \'{}\' not recognized".format(cfg['model']))
    exit(1)



# Perform train/test split
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
dataset_coords = dataset[1]
if ddaBool:
    dataset_labeled = dataset_coords[dataset_coords[:,0] != -1]
else:
    dataset_labeled = dataset_coords[dataset_coords[:,6] != -1]

train_size = int(cfg['train_test_split'] * dataset_labeled.shape[0])
if cfg['train_indices'] == 'None':
        train_indices = np.random.choice(range(np.array(dataset_labeled.shape[0])), train_size, replace=False)
        z=len(train_indices)
        dataset_path = args.config[:-7] + "_%d"%(num_classes)+"_%d"%(z)+"train_indices"
        np.save(dataset_path, train_indices)
        cfg['train_indices'] = dataset_path+'.npy'
        f = open(args.config, 'w')
        f.write(generate_config_silas(cfg))
        f.close()
else:
        train_indices_npy = cfg['train_indices']
        train_indices = np.load(train_indices_npy)
test_indeces = np.setdiff1d(range(np.array(dataset_labeled.shape[0])), train_indices)
#CST20240322 Creating loops so train and test coords aren't 1D
train_coords = []
test_coords = []
for i in train_indices:
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
        transform=transforms.Compose([
    DirectionalVario(vario_num_lag),
    RandomRotateVario(),
])
    )
    valid_dataset = TestDataset(
        imgPath = topDir,
        imgData = dataset_info,
        labels = test_coords,
        train = True,
        transform=transforms.Compose([
    DirectionalVario(vario_num_lag),
    DefaultRotateVario(),
])
        ) 
elif cfg['model'] == 'wri_MLP':
    wri_green = cfg.get('wri_green_band', None)
    wri_red = cfg.get('wri_red_band', None)
    wri_nir = cfg.get('wri_nir_band', None)
    wri_mir = cfg.get('wri_mir_band', None)

    if None in (wri_green, wri_red, wri_nir, wri_mir):
        print("Missing WRI band indices in config. Assuming WV2 bands")
        wri_green = 2
        wri_red = 4
        wri_nir = 6
        wri_mir = 7
    wri_vals = [int(wri_green), int(wri_red), int(wri_nir), int(wri_mir)]
    base = 0 #Index starting at 0
    g_idx, r_idx, nir_idx, mir_idx = [v - base for v in wri_vals]

    stats_bands = sorted(set([g_idx, r_idx, nir_idx, mir_idx]))

    train_dataset = MSPatchStatsDataset(
        imgPath=topDir,
        imgData=dataset_info,
        labels=train_coords,
        wri_bands=(g_idx, r_idx, nir_idx, mir_idx),
        stats_bands=stats_bands,
        train=True
    )

    valid_dataset = MSPatchStatsDataset(
        imgPath=topDir,
        imgData=dataset_info,
        labels=test_coords,
        wri_bands=(g_idx, r_idx, nir_idx, mir_idx),
        stats_bands=stats_bands,
        train=True
    )

    feature_dim = train_dataset.feature_dim
    
    model = PatchMLP(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            activation=activation
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
print(model)
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
date_str = now.strftime("%d-%m-%Y_%H-%M")
config_str = os.path.splitext(os.path.basename(args.config))[0]
output_dir = os.path.join("Output", f"{config_str}_{date_str}")

os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "losses"), exist_ok=True)

print(f"Output saved at {output_dir}")


save_params()

print('----- Training -----')

train_losses = []
valid_losses = []


if cfg['model'] == 'VarioNet':
    for epoch in range(num_epochs):
        sum_loss = 0
        print("EPOCH: {} ".format(epoch),end='',flush=True)
        for batch_idx, (images, variograms, labels) in enumerate(train_loader):
            if args.cuda:
                images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
            images = torch.unsqueeze(images,1).float()
            variograms = torch.unsqueeze(variograms,1).float()
            outputs = model.forward(images, variograms)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
                # Validation phase
        train_losses.append(sum_loss/batch_idx)

        print('running validation')

        loss = 0
        for batch_idx, (images, variograms, labels) in enumerate(valid_loader):
            if args.cuda:
                images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
            images = torch.unsqueeze(images,1).float()
            variograms = torch.unsqueeze(variograms,1).float()
                    
            outputs = model.forward(images, variograms)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
                    #variogram_validation = model(images, variogram=variograms)
                    #variogram_loss = criterion(variogram_validation, labels)
                    #total_variogram_loss += variogram_loss.item()
        valid_losses.append(loss/batch_idx)
        print("\tTRAIN LOSS = {:.5f}\tVALID LOSS = {:.5f}".format(train_losses[-1],valid_losses[-1]))

        checkpoint_str = "epoch_" + str(epoch)


        if valid_losses[-1] == np.array(valid_losses).min():
            checkpoint_path = os.path.join(output_dir, 'checkpoints', checkpoint_str)
            checkpoint = {'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)
        else:
            if len(valid_losses) > np.array(valid_losses).argmin() + 100:
                break
    for param in model.parameters():
            param.requires_grad = True

                # Fine-tune with a smaller learning rate
    optimizer_finetune = torch.optim.Adam(model.parameters(), lr= 5e-6)

                # Fine-tuning loop
    print("Fine Tuning VarioNet")
        

    for epoch in range(fine_epochs):
            sum_loss = 0
            print("EPOCH: {} ".format(epoch),end='',flush=True)
            for batch_idx, (images, variograms, labels) in enumerate(train_loader):
                if args.cuda:
                    images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                images = torch.unsqueeze(images,1).float()
                variograms = torch.unsqueeze(variograms,1).float()
                outputs = model.forward(images, variograms)
                loss = criterion(outputs, labels)

                optimizer_finetune.zero_grad()
                loss.backward()
                optimizer_finetune.step()
                sum_loss += loss.item()
            
            train_losses.append(sum_loss/batch_idx)

                # Validation phase
            print('running validation')
            loss = 0
            for batch_idx, (images, variograms, labels) in enumerate(valid_loader):
                if args.cuda:
                    images, variograms, labels = images.to(device), variograms.to(device), labels.to(device)
                images = torch.unsqueeze(images,1).float()
                variograms = torch.unsqueeze(variograms,1).float()
                    
                outputs = model.forward(images, variograms)
                batch_loss = criterion(outputs, labels)
                loss += batch_loss.item()
                        #variogram_validation = model(images, variogram=variograms)
                        #variogram_loss = criterion(variogram_validation, labels)
                        #total_variogram_loss += variogram_loss.item()
            valid_losses.append(loss/batch_idx)

            print("\tTRAIN LOSS = {:.5f}\tVALID LOSS = {:.5f}".format(train_losses[-1],valid_losses[-1]))

            print('saving checkpoint')
                # Save checkpoint
            checkpoint_str = "epoch_" + str(int(epoch)+int(num_epochs))


            if valid_losses[-1] == np.array(valid_losses).min():
                checkpoint_path = os.path.join(output_dir, 'checkpoints', checkpoint_str)
                checkpoint = {'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict()}
                torch.save(checkpoint, checkpoint_path)
            else:
                if len(valid_losses) > np.array(valid_losses).argmin() + 100:
                    break
# X = tensor
# Y = label
else:
    for epoch in range(num_epochs):

        print("EPOCH: {} ".format(epoch),end='',flush=True)

        sum_loss = 0
        for batch_idx,(X,Y) in enumerate(train_loader):
            
            #CST20240315 else make program exit and say train data set is too low
            if int((len(train_dataset) / batch_size)/10) != 0: #So it won't crash 
                if batch_idx % int((len(train_dataset) / batch_size)/10) == 0:
                    print('.', end='',flush=True)
            else:
                print("ERROR: The length of the training dataset is too small") #CST 20240318
                sys.exit(0)
            
            if cfg['model'] == 'wri_MLP':
                X = X.float()
                Y = Y.long().view(-1)
            else:
                X = torch.unsqueeze(X,1).float()

            # Move batch to GPU
            if args.cuda:
                X,Y = X.to(device),Y.to(device)
                #X = X.view((X.shape[0],1,-1)).float()

            # Compute forward pass
            Y_hat = model.forward(X)

            # Calculate training loss
            loss = criterion(Y_hat, Y)

            # Perform backprop and zero gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sum_loss += loss.item()

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
            if cfg['model'] == 'wri_MLP':
                X = X.float()
                Y = Y.long().view(-1)
            else:
                X = torch.unsqueeze(X,1).float()

            if args.cuda:
                X,Y = X.to(device),Y.to(device)
                #X = X.view((X.shape[0],1,-1)).float()

            # Compute forward pass
            Y_hat = model.forward(X)

            # Calculate training loss
            batch_loss = criterion(Y_hat, Y)
            loss += batch_loss.item()

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
#smallLoss = min(valid_dataset)
checkpoint_path = os.path.join(output_dir, 'checkpoints', checkpoint_str)
checkpoint = {'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}
torch.save(checkpoint, checkpoint_path)
save_losses(checkpoint_str)





"""
# Multispectral training
print("----- Multispectral Training -----")
ms_class_enum = cfg.get('class_enum_MS', [])
if len(ms_class_enum) == 0:
    print("No MS class enumeration found; skipping multispectral training.")
else:
    ms_label_col = 7
    ms_conf_col = 8
    num_classes_ms = len(ms_class_enum)
    ms_model = str(cfg.get('ms_model', 'resnet18')).lower()
    ms_skip = False

    dataset = np.load(dataset_path, allow_pickle=True)
    dataset_info = dataset[0]
    dataset_coords = dataset[1]
    dataset_labeled = dataset_coords[dataset_coords[:,ms_label_col] != -1]

    if dataset_labeled.shape[0] == 0:
        print("No MS labeled data found; skipping multispectral training.")
    else:
        ms_train_size = int(cfg['train_test_split'] * dataset_labeled.shape[0])
        ms_train_indices = np.random.choice(range(np.array(dataset_labeled.shape[0])), ms_train_size, replace=False)
        ms_test_indices = np.setdiff1d(range(np.array(dataset_labeled.shape[0])), ms_train_indices)

        ms_train_coords = []
        ms_test_coords = []
        for i in ms_train_indices:
            ms_train_coords.append(dataset_labeled[i])
        for i in ms_test_indices:
            ms_test_coords.append(dataset_labeled[[i]])

        if ms_model in ("mlp_wri", "wri_mlp", "mlp"):
            wri_green = cfg.get('wri_green_band', None)
            wri_red = cfg.get('wri_red_band', None)
            wri_nir = cfg.get('wri_nir_band', None)
            wri_mir = cfg.get('wri_mir_band', None)

            if None in (wri_green, wri_red, wri_nir, wri_mir):
                print("Missing WRI band indices in config; skipping MS training.")
                ms_skip = True
            if not ms_skip:
                wri_vals = [int(wri_green), int(wri_red), int(wri_nir), int(wri_mir)]
                base = 1 if min(wri_vals) >= 1 else 0
                g_idx, r_idx, nir_idx, mir_idx = [v - base for v in wri_vals]

                stats_bands = sorted(set([g_idx, r_idx, nir_idx, mir_idx]))

                train_dataset_ms = MSPatchStatsDataset(
                    imgPath=topDir,
                    imgData=dataset_info,
                    labels=ms_train_coords,
                    wri_bands=(g_idx, r_idx, nir_idx, mir_idx),
                    stats_bands=stats_bands,
                    train=True,
                    label_col=ms_label_col,
                    conf_col=ms_conf_col,
                )

                valid_dataset_ms = MSPatchStatsDataset(
                    imgPath=topDir,
                    imgData=dataset_info,
                    labels=ms_test_coords,
                    wri_bands=(g_idx, r_idx, nir_idx, mir_idx),
                    stats_bands=stats_bands,
                    train=True,
                    label_col=ms_label_col,
                    conf_col=ms_conf_col,
                )

                train_loader_ms = DataLoader(train_dataset_ms, batch_size=batch_size, shuffle=True)
                valid_loader_ms = DataLoader(valid_dataset_ms, batch_size=1, shuffle=False)

                feature_dim = train_dataset_ms.feature_dim
                if feature_dim == 0:
                    print("No MS features found; skipping MS training.")
                    ms_skip = True
                else:
                    ms_hidden_layers = cfg.get('ms_hidden_layers', [32, 16])
                    ms_activation = cfg.get('ms_activation', 'ReLU')
                    ms_dropout = cfg.get('ms_dropout', 0.0)
                    model_ms = PatchMLP(
                        input_dim=feature_dim,
                        num_classes=num_classes_ms,
                        hidden_layers=ms_hidden_layers,
                        activation=ms_activation,
                        dropout=ms_dropout,
                    )

                    criterion_ms = torch.nn.CrossEntropyLoss()
                    optimizer_ms = optim.Adam(model_ms.parameters(), lr=learning_rate)

                    device_ms = torch.device("cuda:0") if args.cuda else torch.device("cpu")
                    if args.cuda:
                        model_ms.cuda()
        else:
            model_ms = Resnet18.resnet18(pretrained=False, num_classes=num_classes_ms)
            rgb_bands = cfg['bands']

            def ms_transform(img):
                # img: (bands, H, W) uint8 -> (3, 224, 224) float32
                img = img[rgb_bands, :, :]
                img = torch.from_numpy(img).float() / 255.0
                img = TF.resize(img, (224, 224))
                return img

            train_dataset_ms = SplitImageDatasetMS(
                imgPath = topDir,
                imgData = dataset_info,
                labels = ms_train_coords,
                train = True,
                transform = ms_transform,
                label_col = ms_label_col,
                conf_col = ms_conf_col
                )

            valid_dataset_ms = SplitImageDatasetMS(
                imgPath = topDir,
                imgData = dataset_info,
                labels = ms_test_coords,
                train = True,
                transform = ms_transform,
                label_col = ms_label_col,
                conf_col = ms_conf_col
                )

            train_loader_ms = DataLoader(train_dataset_ms, batch_size=batch_size, shuffle=True)
            valid_loader_ms = DataLoader(valid_dataset_ms, batch_size=1, shuffle=False)

            criterion_ms = torch.nn.CrossEntropyLoss()
            optimizer_ms = optim.Adam(model_ms.parameters(),lr=learning_rate)

            device_ms = torch.device("cuda:0") if args.cuda else torch.device("cpu")
            if args.cuda:
                model_ms.cuda()

        if not ms_skip:
            # Separate output directory for MS model
            output_dir_ms = output_dir + "_ms"
            os.makedirs(os.path.join(output_dir_ms, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(output_dir_ms, "losses"), exist_ok=True)

            train_losses_ms = []
            valid_losses_ms = []

            for epoch in range(num_epochs):
                print("MS EPOCH: {} ".format(epoch),end='',flush=True)
                sum_loss = 0
                for batch_idx,(X,Y) in enumerate(train_loader_ms):
                    X,Y = X.to(device_ms),Y.to(device_ms)
                    X = X.float()
                    Y_hat = model_ms.forward(X)
                    loss = criterion_ms(Y_hat, Y)
                    optimizer_ms.zero_grad()
                    loss.backward()
                    optimizer_ms.step()
                    sum_loss += loss.item()

                train_losses_ms.append(sum_loss/max(batch_idx,1))

                # Validation
                loss = 0
                for batch_idx,(X,Y) in enumerate(valid_loader_ms):
                    X,Y = X.to(device_ms),Y.to(device_ms)
                    X = X.float()
                    Y_hat = model_ms.forward(X)
                    batch_loss = criterion_ms(Y_hat, Y)
                    loss += batch_loss.item()

                valid_losses_ms.append(loss/max(batch_idx,1))
                print("\tMS TRAIN LOSS = {:.5f}\tMS VALID LOSS = {:.5f}".format(train_losses_ms[-1],valid_losses_ms[-1]))

                checkpoint_str_ms = "epoch_" + str(epoch)
                if valid_losses_ms[-1] == np.array(valid_losses_ms).min():
                    checkpoint_path_ms = os.path.join(output_dir_ms, 'checkpoints', checkpoint_str_ms)
                    checkpoint_ms = {'state_dict': model_ms.state_dict(),
                                    'optimizer' : optimizer_ms.state_dict()}
                    torch.save(checkpoint_ms, checkpoint_path_ms)

            # Save loss curves
            np.save(os.path.join(output_dir_ms, "losses", "ms_train_losses"), np.array(train_losses_ms))
            np.save(os.path.join(output_dir_ms, "losses", "ms_valid_losses"), np.array(valid_losses_ms))

"""
