from torch.utils.data import DataLoader
import torch
from Dataset_MS import *
from Dataset_MS import MSPatchStatsDataset
from torch import optim
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import random
from Models import *
from Models.patchMLP import PatchMLP
from VarioNet_MS import *
import yaml
import warnings
from sklearn.utils.class_weight import compute_class_weight

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--load_checkpoint_ms", type=str, default=None)
parser.add_argument("--netCDF", action="store_true")
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
adapt = cfg['adaptive']

patch_size = cfg.get('patch_size', (50, 50))

# Set dataset hyperparameters as specified by config file
topDir = cfg['img_path']
classEnumMS = cfg.get('class_enum_MS', cfg.get('class_enum_MS'))
classEnumPan = cfg.get('class_enum_PAN', cfg.get('class_enum_PAN'))
dataset_path = cfg['npy_path']
train_path = cfg['train_path']
valid_path = cfg['valid_path']

# Initialize NN model as specified by config file
print('----- Initializing Panchromatic Neural Network Model -----')
ddaBool = False
if cfg['PAN_model'] == 'VarioMLP':
    PANnum_classes = len(cfg['class_enum_PAN'])
    vario_num_lag = cfg['vario_num_lag']
    hidden_layers = cfg['hidden_layers']
    imSize = cfg['split_img_size']
    PANmodel = VarioMLP.VarioMLP(PANnum_classes, vario_num_lag, hidden_layers=hidden_layers) 
    img_transforms_valid = transforms.Compose([
        DirectionalVario(PANmodel.num_lag),
        DefaultRotateVario(),
    ])
elif cfg['PAN_model'] == 'Resnet18':
    PANnum_classes = len(cfg['class_enum_PAN'])
    PANmodel = Resnet18.resnet18(pretrained=False, num_classes=PANnum_classes)
    img_transforms_valid = None
elif cfg['PAN_model'] == 'VarioNet':
    PANnum_classes = len(cfg['class_enum_PAN'])
    vario_num_lag = cfg['vario_num_lag']
    image_folder = cfg['training_img_path']
    alpha = cfg['alpha']
    beta = cfg['beta']
    vario_mlp = VarioMLP.VarioMLP(PANnum_classes, vario_num_lag, hidden_layers=hidden_layers)
    resnet18 = Resnet18.resnet18(pretrained=False, num_classes=PANnum_classes)
    vario_mlp.load_state_dict(torch.load('vario_mlp.pth'))
    resnet18.load_state_dict(torch.load('resnet18.pth'))
    PANmodel = CombinedModel(vario_mlp, resnet18, PANnum_classes, a = alpha, b = beta, adaptive=adapt)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
        ])

elif cfg['PAN_model'] == 'DDAiceNet':
    ddaBool = True
    PANnum_classes = len(cfg['class_enum_PAN'])
    nres = cfg['nres']
    hidden_layers = cfg['hidden_layers']
    PANmodel = DDAiceNet.DDAiceNet(PANnum_classes, nres*2, hiddenLayers=hidden_layers)
    img_transforms_valid = None
else:
    print("Error: Model \'%s\' not recognized"%(cfg['PAN_model']))
    exit(1)

print(PANmodel)

# Load model checkpoint
if args.load_checkpoint:
    PANcheckpoint_path = args.load_checkpoint
    PANcheckpoint_str = os.path.basename(PANcheckpoint_path)
    output_dir = os.path.dirname(os.path.dirname(PANcheckpoint_path))
    print(PANcheckpoint_str, output_dir)
    PANcheckpoint = torch.load(args.load_checkpoint)
    PANmodel.load_state_dict(PANcheckpoint['state_dict'])

else:
    print("Please specify a model checkpoint with the --load_checkpoint argument")
    exit(1)

# Initialize Datasets and DataLoaders
print('----- Initializing Dataset -----')
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
dataset_labels = dataset[1]

if cfg['PAN_model'] == 'VarioMLP' or cfg['PAN_model'] == 'Resnet18':
    valid_dataset = SplitImageDatasetPAN(
        imgPath = topDir,
        imgData = dataset_info,
        labels = dataset_labels,
        train = False,
        transform = img_transforms_valid
        )
elif cfg['PAN_model'] == 'VarioNet':
    #Currently does not work for, need to adjust TestDataset to work for the indices
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

print('----- Initializing Panchromatic DataLoader -----')

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
        optimizer = optim.Adam(PANmodel.parameters(),lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
else:
    # Initialize loss critereron and gradient descent optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(PANmodel.parameters(),lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

# Initialize cuda
if args.cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    PANmodel.cuda()
    #optimizer.cuda()

# Constants from Authors100 dataset
MAXWIDTH = 2260
MAXHEIGHT = 337

softmax = torch.nn.Softmax(dim=1)

print('----- Initializing DataLoader -----')


print('Done!')

print('----- Training Panchromatic-----')

labels = []
confs = []


if cfg['PAN_model'] == 'VarioNet':

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
        Y_hat = PANmodel.forward(images, variograms)
            # Apply softmax to get probabilities
        sm = softmax(Y_hat)

            # Get the max confidence score and corresponding label
        conf = sm.max()

        if conf > 0:
                labels.append(int(torch.argmax(Y_hat)))
                confs.append(conf.item())
        else:
                labels.append(PANnum_classes)
else:              
    for batch_idx,X in enumerate(valid_loader):

        if batch_idx % 100 == 0:
            print(f"Processing batch {batch_idx}")
                
        # Move batch to GPU
        if args.cuda:
            X = X.to(device)

        X = torch.unsqueeze(X,1).float()

        # Compute forward pass
        Y_hat = PANmodel.forward(X)

        sm = softmax(Y_hat)

        conf = sm.max()

        if conf > 0:
            labels.append(int(torch.argmax(Y_hat)))
            confs.append(conf.item())
        else:
            labels.append(PANnum_classes)

split_info = dataset[1]
if ddaBool:
    split_info[:,0] = labels
    split_info[:,1] = confs
else:
    split_info[:,6] = labels
    split_info[:,7] = confs
#split_info = np.concatenate((split_info, np.array(confs).reshape(len(confs),1)),1)

print(dataset[1].shape)
dataset[1] = split_info
#data.append(confs)
np.save(output_dir+"/labels/labeled_"+PANcheckpoint_str, dataset)

if adapt and cfg['PAN_model'] == 'VarioNet':
     PANmodel.plot_beta(output_dir=output_dir, conf = confs)

if args.netCDF:
    to_netCDF(dataset)

"""

# Initialize NN model as specified by config file
print('----- Initializing Multi-Spectral Neural Network Model -----')
ddaBool = False
if cfg['MS_model'] == 'patchMLP':
    num_classes = len(classEnumMS)
    vario_num_lag = cfg['vario_num_lag']
    hidden_layers = cfg['hidden_layers']
    imSize = cfg['split_img_size']
    image_folder = cfg['training_img_path']
    activation = cfg['activation']
    #Dropout is currently hardcoded to be 0 within PatchMLP
    img_transforms_valid = None
elif cfg['MS_model'] == 'msCNN':
    num_classes = len(classEnumMS)
    image_folder = cfg['training_img_path']
    model_ms = patchMLP.WVSpecPatchCNN(in_channels=8, num_classes=num_classes)
    img_transforms_train = None
    img_transforms_valid = None
elif cfg['MS_model'] == 'Resnet18':
    num_classes = len(classEnumMS)
    model_ms = Resnet18.resnet18(pretrained=False, num_classes=num_classes, inchannels=8)
    img_transforms_valid = None

else:
    print("Error: Model \'%s\' not recognized"%(cfg['MS_model']))
    exit(1)



# Initialize Datasets and DataLoaders
print('----- Initializing Dataset (Multi-Spectral) -----')


if cfg['MS_model'] == 'Resnet18' or cfg['MS_model'] == 'msCNN':
    valid_dataset = SplitImageDatasetMS(
        imgPath = topDir,
        imgData = dataset_info,
        labels = dataset_labels,
        train = False,
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
    

    # Feature dim: 8 bands * 3 (mean+std+entropy) + 4 (NDWI and NDSI mean+std) + 1 (brightness) = 77
    feature_dim = 4 * 2 + 6
    
    model_ms = PatchMLP(
        input_dim=feature_dim,
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
    valid_dataset = DDAiceDataset(
        dataPath = topDir,
        dataInfo = dataset_info,
        dataLabeled = dataset_labels,
        train = False,
        transform = None
        )

print(model_ms)

print('Test set size: \t%d images'%(len(valid_dataset)))

print('----- Initializing DataLoader -----')

valid_loader_ms = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False
)

# For patchMLP, compute actual feature dimension from first batch before loading checkpoint
if cfg['MS_model'] == 'patchMLP' and args.load_checkpoint_ms:
    first_batch = next(iter(valid_loader_ms))
    # Handle both (X, Y) and (X,) return formats
    if isinstance(first_batch, (list, tuple)):
        first_batch_X = first_batch[0]
    else:
        first_batch_X = first_batch
    from utils_MS import compute_patch_wri_features_batch
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
    print("Model rebuilt with correct input_dim. Architecture after rebuild:")
    print(model_ms)

# Load model checkpoint
if args.load_checkpoint_ms:
    checkpoint_path_ms = args.load_checkpoint_ms
    checkpoint_str_ms = os.path.basename(checkpoint_path_ms)
    output_dir = os.path.dirname(os.path.dirname(checkpoint_path_ms))
    print(checkpoint_str_ms, output_dir)
    ms_checkpoint = torch.load(args.load_checkpoint_ms)
    model_ms.load_state_dict(ms_checkpoint['state_dict'])
else:
    print("Please specify a model checkpoint with the --load_checkpoint_ms argument")
    exit(1)


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
        criterion_ms = torch.nn.CrossEntropyLoss(weight=class_wts)
        optimizer_ms = optim.Adam(model_ms.parameters(),lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ms, gamma=0.9)
else:
    # Initialize loss critereron and gradient descent optimizer
    criterion_ms = torch.nn.CrossEntropyLoss()
    optimizer_ms = optim.Adam(model_ms.parameters(),lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

# Initialize cuda
if args.cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model_ms.cuda()
    #optimizer.cuda()


softmax = torch.nn.Softmax(dim=1)


print('Done!')

print('----- Training Multi-Spectral -----')

# MS testing

if args.cuda:
    model_ms.cuda()

labels_ms = []
confs_ms = []

if cfg['MS_model'] == 'patchMLP':
    iter_loader = ((x, None) for x in valid_loader_ms)
else:
    iter_loader = ((x, None) for x in valid_loader_ms)

for batch_idx, (X_ms, Y_ms) in enumerate(iter_loader):
    if batch_idx % 100 == 0:
        print(f"Processing MS batch {batch_idx}")

    if args.cuda:
        X_ms = X_ms.to(device)

    X_ms = X_ms.float()
    
    # Compute features if using wriMLP
    if cfg['MS_model'] == 'patchMLP':
        from utils_MS import compute_patch_wri_features_batch
        #patch_size = cfg.get('patch_size', (32, 32))
        X_ms = compute_patch_wri_features_batch(X_ms, patch_size, wri_config)  # (1, F, Hp, Wp)
        X_ms = X_ms.reshape(X_ms.shape[0], -1)  # flatten to (1, F*Hp*Wp)
        if args.cuda:
            X_ms = X_ms.to(device)
    
    if cfg['MS_model'] == 'msCNN':
        logits = model_ms(X_ms)                 # [1, C]
        sm_ms = softmax(logits)
        conf_ms = sm_ms.max()
        labels_ms.append(int(torch.argmax(logits)))
        confs_ms.append(conf_ms.item())
    else:
        Y_hat_ms = model_ms.forward(X_ms)
        sm_ms = softmax(Y_hat_ms)
        conf_ms = sm_ms.max()

        if conf_ms > 0:
            labels_ms.append(int(torch.argmax(Y_hat_ms)))
            confs_ms.append(conf_ms.item())
        else:
            labels_ms.append(num_classes)

split_info_ms = dataset[1]
labels_arr = np.array(labels_ms)
confs_arr = np.array(confs_ms)



ms_mask = split_info_ms[:,2] >= 0

# Reconstruct which MS split rows produced non-empty patches (same order as the dataset)
winSize = dataset_info.get('MS_winsize_pix', None)
if winSize is None:
    print('Warning: MS winSize not found in dataset info; falling back to assigning by ms mask')

# Prefer MS image paths from the saved dataset info (this matches dataset creation order)
ms_image_paths = []
filenames = dataset_info.get('filename', None)
if filenames is not None:
    for p in filenames:
        try:
            with rio.open(p) as ds:
                if ds.count and ds.count > 1:
                    ms_image_paths.append(p)
        except Exception:
            # ignore images we cannot open
            continue
else:
    # Fallback: scan filesystem similar to Dataset_MS.getImgPathsMS
    scene_dirs = sorted([d for d in __import__('glob').glob(os.path.join(topDir, '*')) if os.path.isdir(d)])
    for scene in scene_dirs:
        tifs = sorted(__import__('glob').glob(os.path.join(scene, '*.tif')))
        for p in tifs:
            try:
                with rio.open(p) as ds:
                    if ds.count and ds.count > 1:
                        ms_image_paths.append(p)
            except Exception:
                continue

valid_src_indices = []
if winSize is not None:
    # If dataset_info provides the original filename ordering, use it to map stored img_num
    filenames = dataset_info.get('filename', None)
    if filenames is not None:
        # Build mapping from ms_image enumerate index -> file index in filenames
        ms_file_indices = []
        for p in ms_image_paths:
            try:
                ms_file_indices.append(filenames.index(p))
            except ValueError:
                # If path not found (e.g., different path formats), try basename match
                bas = os.path.basename(p)
                found = -1
                for k,fp in enumerate(filenames):
                    if os.path.basename(fp) == bas:
                        found = k
                        break
                if found >= 0:
                    ms_file_indices.append(found)
                else:
                    ms_file_indices.append(None)

    for imgNum, imgPath in enumerate(ms_image_paths):
        try:
            with rio.open(imgPath) as ds:
                height = ds.height
                width = ds.width
        except Exception:
            continue

        # Determine which file-index was used when the dataset was created
        if 'ms_file_indices' in locals() and ms_file_indices[imgNum] is not None:
            file_idx = ms_file_indices[imgNum]
            rows_idx = np.where(split_info_ms[:, 10] == file_idx)[0]
        else:
            # fallback: try matching by imgNum itself
            rows_idx = np.where(split_info_ms[:, 10] == imgNum)[0]

        for ridx in rows_idx:
            x = int(split_info_ms[ridx, 2])
            y = int(split_info_ms[ridx, 3])
            if x < 0 or y < 0:
                continue
            # check patch fits inside image
            if (x + winSize[0] > height) or (y + winSize[1] > width):
                continue
            valid_src_indices.append(ridx)

    # Align predictions to valid source indices without using row_indices
    n_preds = len(labels_arr)
    n_valid = len(valid_src_indices)
    if n_preds == 0:
        print('No MS predictions to write; skipping assignment.')
    else:
        if n_preds != n_valid:
            print(f"Warning: number of rows ({len(n_preds)}) != number of predictions ({len(n_valid)}). Aligning to min length.")
        n = min(n_preds, n_valid)
        if n > 0:
            src = valid_src_indices[:n]
            split_info_ms[src, 8] = labels_arr[:n]
            split_info_ms[src, 9] = confs_arr[:n]
        else:
            print('No overlapping valid patches and predictions; skipping assignment.')
else:
    # Fallback: assign to rows that appear to be MS (ms_x >= 0)
    ms_indices = np.where(ms_mask)[0]
    print('ms rows in split table (fallback):', ms_indices.shape[0])
    if len(ms_indices) != len(labels_arr):
        print(f"Warning: number of rows ({len(ms_indices)}) != number of predictions ({len(labels_arr)}). Aligning to min length.")
        n = min(len(ms_indices), len(labels_arr))
        if n > 0:
            split_info_ms[ms_indices[:n], 8] = labels_arr[:n]
            split_info_ms[ms_indices[:n], 9] = confs_arr[:n]
        else:
            print('No MS rows or no MS predictions; skipping assignment.')
    else:
        split_info_ms[ms_indices, 8] = labels_arr
        split_info_ms[ms_indices, 9] = confs_arr


print(dataset[1].shape)
dataset[1] = split_info_ms

ms_checkpoint_str = os.path.basename(args.load_checkpoint_ms)
ms_output_dir = os.path.dirname(os.path.dirname(args.load_checkpoint_ms))
os.makedirs(os.path.join(ms_output_dir, "labels"), exist_ok=True)
np.save(os.path.join(ms_output_dir, "labels", "labeled_"+ms_checkpoint_str), dataset)

if args.netCDF:
    to_netCDF(dataset)

"""