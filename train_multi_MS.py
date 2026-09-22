from xml.parsers.expat import model

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
from sklearn.model_selection import train_test_split



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
patch_size = cfg.get('patch_size', (50, 50))  # default to (8,8) if not in config

# Set dataset hyperparameters as specified by config file
topDir = cfg['img_path']
classEnum = cfg.get('class_enum_MS', cfg.get('class_enum_MS'))
dataset_path = cfg['npy_path']
train_path = cfg['train_path']
valid_path = cfg['valid_path']

# Initialize NN model as specified by config file
print('----- Initializing Neural Network Model -----')
#initializing ddaBool
ddaBool = False
if cfg['MS_model'] == 'patchMLP':
    num_classes = len(classEnum)
    vario_num_lag = cfg['vario_num_lag']
    hidden_layers = cfg['hidden_layers']
    imSize = cfg['split_img_size']
    image_folder = cfg['training_img_path']
    activation = cfg['activation']
    #Dropout is currently hardcoded to be 0 within PatchMLP
    img_transforms_train = None
    img_transforms_valid = None

elif cfg['MS_model'] == 'msCNN':
    num_classes = len(classEnum)
    image_folder = cfg['training_img_path']
    model_ms = patchMLP.WVSpecPatchCNN(in_channels=8, num_classes=num_classes)
    img_transforms_train = None
    img_transforms_valid = None
elif cfg['MS_model'] == 'Resnet18':
    num_classes = len(classEnum)
    image_folder = cfg['training_img_path']
    model_ms = Resnet18.resnet18(pretrained=False, num_classes=num_classes, inchannels=8)
    """transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
        ])
    """    
    img_transforms_train = None
    img_transforms_valid = None

else:
    print("Error: Model \'{}\' not recognized".format(cfg['MS_model']))
    exit(1)



# Perform train/test split
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
dataset_coords = dataset[1]
if ddaBool:
    dataset_labeled = dataset_coords[dataset_coords[:,0] != -1]
else:
    dataset_labeled = dataset_coords[dataset_coords[:,8] != -1]

train_size = int(cfg['train_test_split'] * dataset_labeled.shape[0])


if cfg['train_indices'] == 'None':
    all_idx = np.arange(dataset_labeled.shape[0])

    # class labels for stratification
    labels = dataset_labeled[:, 8].astype(int)   # for MS use column 8
    # if using DDA, change to the right label column, e.g. dataset_labeled[:, 4]

    train_indices, _ = train_test_split(
        all_idx,
        train_size=train_size,
        stratify=labels,
        random_state=cfg.get('random_seed', 42)
    )

    z = len(train_indices)
    dataset_path = args.config[:-7] + "_%d"%(num_classes)+"_%d"%(z)+"train_indices"
    np.save(dataset_path, train_indices)
    cfg['train_indices'] = dataset_path + '.npy'
    with open(args.config, 'w') as f:
        f.write(generate_config_silas(cfg))
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

if cfg['MS_model'] == 'Resnet18' or cfg['MS_model'] == 'msCNN':
    train_dataset = SplitImageDatasetMS(
        imgPath = topDir,
        imgData = dataset_info,
        labels = train_coords,
        train = True,
        transform = img_transforms_train
        )

    valid_dataset = SplitImageDatasetMS(
        imgPath = topDir,
        imgData = dataset_info,
        labels = test_coords,
        train = True,
        transform = img_transforms_valid
        )

elif cfg['MS_model'] == 'patchMLP':
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
    base = 0
    g_idx, r_idx, nir_idx, mir_idx = [v - base for v in wri_vals]

    stats_bands = sorted(set([g_idx, r_idx, nir_idx, mir_idx]))
    #stats_bands = list(range(8))

    # Use SplitImageDatasetMS for wriMLP too (same as ResNet)
    train_dataset = SplitImageDatasetMS(
        imgPath=topDir,
        imgData=dataset_info,
        labels=train_coords,
        train=True
    )

    valid_dataset = SplitImageDatasetMS(
        imgPath=topDir,
        imgData=dataset_info,
        labels=test_coords,
        train=True
    )

    # Feature dim will be computed dynamically from first batch
    # For now use a placeholder; will be verified on first forward pass
    feature_dim_placeholder = 4 * 2 + 6  # fallback
    
    model_ms = PatchMLP(
        input_dim=feature_dim_placeholder,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        activation=activation
    )
    
    # Store WRI band indices for feature computation in the loop
    wri_config = {
        'g_idx': g_idx, 'r_idx': r_idx, 'nir_idx': nir_idx, 'mir_idx': mir_idx,
        'stats_bands': stats_bands, 'eps': 1e-6
    }

        
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
print(model_ms)
#CST20240315
print('Training set size: \t%d images'%(len(train_dataset)))
# for i in range(num_classes):
#     print('Class {}: {} - {} train images'.format(i,classEnum[i],len(train_coords[train_coords[:,4] == i])))
# print('Validation set size: \t%d images'%(len(valid_dataset)))
# for i in range(num_classes):
#     print('Class {}: {} - {} valid images'.format(i,classEnum[i],len(test_coords[test_coords[:,4] == i])))
print('----- Initializing DataLoader -----')

train_loader_ms = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
    )
print("train loader", type(train_loader_ms))
valid_loader_ms = DataLoader(
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
        criterion_ms = torch.nn.CrossEntropyLoss(weight=class_wts)
        optimizer_ms = optim.Adam(model_ms.parameters(),lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ms, gamma=0.9) #TODO: what this does?
else:
    # Initialize loss critereron and gradient descent optimizer
    criterion_ms = torch.nn.CrossEntropyLoss()
    optimizer_ms = optim.Adam(model_ms.parameters(),lr=learning_rate)

# Load model checkpoint
if args.load_checkpoint:
    checkpoint_path = args.load_checkpoint
    checkpoint = torch.load(checkpoint_path)
    model_ms.load_state_dict(checkpoint['state_dict'])

# Initialize cuda
if args.cuda:
    print('----- Initializing CUDA -----')
    torch.cuda.set_device(0)
    device_ms = torch.device("cuda:0")
    model_ms.cuda()
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


# Multispectral training
print("----- Multispectral Training -----")
ms_class_enum = cfg.get('class_enum_MS', [])
if len(ms_class_enum) == 0:
    print("No MS class enumeration found; skipping multispectral training.")
else:
        # Get actual feature dimension from first batch before training
        first_batch_X, _ = next(iter(train_loader_ms))
        if cfg['MS_model'] == 'patchMLP':
            if args.cuda:
                first_batch_X = first_batch_X.to(device_ms)
            with torch.no_grad():
                feats_sample = compute_patch_wri_features_batch(first_batch_X, patch_size, wri_config)
                actual_feature_dim = feats_sample.shape[1] * feats_sample.shape[2] * feats_sample.shape[3]
            print(f'[Info] Actual feature dimension from first batch: {actual_feature_dim}')
            # Rebuild model with correct input_dim
            model_ms = PatchMLP(
                input_dim=actual_feature_dim,
                num_classes=num_classes,
                hidden_layers=hidden_layers,
                activation=activation
            )
            if args.cuda:
                model_ms.cuda()
            criterion_ms = torch.nn.CrossEntropyLoss()
            optimizer_ms = optim.Adam(model_ms.parameters(), lr=learning_rate)
        
        # Separate output directory for MS model
        output_dir_ms = output_dir + "_ms"
        os.makedirs(os.path.join(output_dir_ms, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir_ms, "losses"), exist_ok=True)

        train_losses_ms = []
        valid_losses_ms = []

        for epoch in range(num_epochs):
            print(f"MS EPOCH: {epoch}", end='', flush=True)
            sum_loss = 0

            for batch_idx, (X, Y) in enumerate(train_loader_ms):
                if args.cuda:
                    X, Y = X.to(device_ms), Y.to(device_ms)

                X = X.float()  # Convert from uint8 to float32

                if cfg['MS_model'] == 'patchMLP':
                    X = compute_patch_wri_features_batch(X, patch_size, wri_config)  # shape [B, F, Hp, Wp]
                    X = X.reshape(X.shape[0], -1)  # flatten to [B, F*Hp*Wp]
                    Y_hat = model_ms.forward(X)
                    loss = criterion_ms(Y_hat, Y)
                elif cfg['MS_model'] == 'msCNN':
                    logits = model_ms(X)  # (B, num_classes)
                    loss = criterion_ms(logits, Y)
                else:
                    Y_hat = model_ms.forward(X)
                    loss = criterion_ms(Y_hat, Y)

                optimizer_ms.zero_grad()
                loss.backward()
                optimizer_ms.step()

                sum_loss += loss.item()

            train_losses_ms.append(sum_loss / max(batch_idx, 1))

            # validation loop
            val_loss = 0
            for batch_idx, (X, Y) in enumerate(valid_loader_ms):
                if args.cuda:
                    X, Y = X.to(device_ms), Y.to(device_ms)

                X = X.float()  # Convert from uint8 to float32

                with torch.no_grad():
                    if cfg['MS_model'] == 'patchMLP':
                        X = compute_patch_wri_features_batch(X, patch_size, wri_config)  # shape [B, F, Hp, Wp]
                        X = X.reshape(X.shape[0], -1)  # flatten to [B, F*Hp*Wp]
                        Y_hat = model_ms.forward(X)
                        val_loss += criterion_ms(Y_hat, Y).item()
                    elif cfg['MS_model'] == 'msCNN':
                        logits = model_ms(X)  # (B, num_classes)
                        val_loss += criterion_ms(logits, Y).item()
                    else:
                        Y_hat = model_ms.forward(X)
                        val_loss += criterion_ms(Y_hat, Y).item()

            valid_losses_ms.append(val_loss / max(batch_idx, 1))
            print(f"\tMS TRAIN LOSS = {train_losses_ms[-1]:.5f}\tMS VALID LOSS = {valid_losses_ms[-1]:.5f}")

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                model_ms.eval()
                with torch.no_grad():
                    x_val, y_val = next(iter(valid_loader_ms))  # grab a batch
                    if args.cuda:
                        x_val, y_val = x_val.to(device_ms), y_val.to(device_ms)
                    x_val = x_val.float()  # Convert from uint8 to float32
                    if cfg['MS_model'] == 'msCNN':
                        logits = model_ms(x_val)
                        preds = logits.argmax(dim=1)
                        import matplotlib.pyplot as plt
                        n_plots = min(4, preds.shape[0])
                        fig, axes = plt.subplots(1, n_plots, figsize=(16, 4))
                        if n_plots == 1:
                            axes = [axes]
                        for i in range(n_plots):
                            axes[i].text(0.5, 0.5, f"Pred: {preds[i].item()}\nTrue: {y_val[i].item()}",
                                         ha='center', va='center', fontsize=16)
                            axes[i].set_axis_off()
                        plt.savefig(os.path.join(output_dir_ms, f'pred_epoch_{epoch}.png'))
                        plt.close()

            checkpoint_str_ms = "epoch_" + str(epoch)
            if valid_losses_ms[-1] == np.array(valid_losses_ms).min():
                checkpoint_path_ms = os.path.join(output_dir_ms, 'checkpoints', checkpoint_str_ms)
                checkpoint_ms = {'state_dict': model_ms.state_dict(),
                                'optimizer' : optimizer_ms.state_dict()}
                torch.save(checkpoint_ms, checkpoint_path_ms)

        # Save loss curves
        np.save(os.path.join(output_dir_ms, "losses", "ms_train_losses"), np.array(train_losses_ms))
        np.save(os.path.join(output_dir_ms, "losses", "ms_valid_losses"), np.array(valid_losses_ms))

