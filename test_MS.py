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
#parser.add_argument("--load_checkpoint_ms", type=str, default=None)
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
elif cfg['model'] == 'wri_MLP':
    num_classes = cfg['num_classes']
    vario_num_lag = cfg['vario_num_lag']
    hidden_layers = cfg['hidden_layers']
    imSize = cfg['split_img_size']
    image_folder = cfg['training_img_path']
    activation = cfg['activation']
    #Dropout is currently hardcoded to be 0 within PatchMLP
    img_transforms_valid = None
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



# Initialize Datasets and DataLoaders
print('----- Initializing Dataset -----')
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
dataset_labels = dataset[1]
allowed_classes = None
if cfg['model'] == 'wri_MLP':
    labels_unique = np.unique(dataset_labels[:, 6]).astype(int)
    allowed_classes = [int(x) for x in labels_unique if x >= 0]
    if len(allowed_classes) == 0:
        allowed_classes = None
    else:
        print(f"WRI MLP: masking predictions to labeled classes {allowed_classes}")

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

    valid_dataset = MSPatchStatsDataset(
        imgPath=topDir,
        imgData=dataset_info,
        labels=dataset_labels,
        wri_bands=(g_idx, r_idx, nir_idx, mir_idx),
        stats_bands=stats_bands,
        train=True
    )

    feature_dim = valid_dataset.feature_dim
    
    model = PatchMLP(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            activation=activation
        )


else:
    valid_dataset = DDAiceDataset(
        dataPath = topDir,
        dataInfo = dataset_info,
        dataLabeled = dataset_labels,
        train = False,
        transform = None
        )

print(model)

# Load model checkpoint
if args.load_checkpoint:
    checkpoint_path = args.load_checkpoint
    checkpoint_str = os.path.basename(checkpoint_path)
    output_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    print(checkpoint_str, output_dir)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

else:
    print("Please specify a model checkpoint with the --load_checkpoint argument")
    exit(1)

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
    if cfg['model'] == 'wri_MLP':
        iter_loader = ((x, y) for x, y in valid_loader)
    else:
        iter_loader = ((x, None) for x in valid_loader)

    for batch_idx,(X, Y) in enumerate(iter_loader):

        if batch_idx % 100 == 0:
            print(f"Processing batch {batch_idx}")

        if cfg['model'] == 'wri_MLP':
            X = X.float()
        else:
            X = torch.unsqueeze(X,1).float()

        # Move batch to GPU
        if args.cuda:
            X = X.to(device)

        # Compute forward pass
        Y_hat = model.forward(X)
        if allowed_classes is not None:
            mask = torch.full((Y_hat.shape[1],), float('-inf'), device=Y_hat.device)
            mask[allowed_classes] = 0.0
            Y_hat = Y_hat + mask

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
    split_info[:,6] = labels
    split_info[:,7] = confs
#split_info = np.concatenate((split_info, np.array(confs).reshape(len(confs),1)),1)

print(dataset[1].shape)
dataset[1] = split_info
#data.append(confs)
np.save(output_dir+"/labels/labeled_"+checkpoint_str, dataset)

if adapt and cfg['model'] == 'VarioNet':
     model.plot_beta(output_dir=output_dir, conf = confs)


"""
# MS testing (WRI MLP)
if args.load_checkpoint_ms:
    ms_model = str(cfg.get('ms_model', 'resnet18')).lower()
    if ms_model not in ("mlp_wri", "wri_mlp", "mlp"):
        print("MS model is not set to WRI MLP; skipping MS testing.")
    else:
        ms_class_enum = cfg.get('class_enum_MS', [])
        if len(ms_class_enum) == 0:
            print("No MS class enumeration found; skipping MS testing.")
        else:
            print('----- MS Testing (WRI MLP) -----')
            ms_label_col = 7
            ms_conf_col = 8
            num_classes_ms = len(ms_class_enum)

            wri_green = cfg.get('wri_green_band', None)
            wri_red = cfg.get('wri_red_band', None)
            wri_nir = cfg.get('wri_nir_band', None)
            wri_mir = cfg.get('wri_mir_band', None)

            if None in (wri_green, wri_red, wri_nir, wri_mir):
                print("Missing WRI band indices in config; skipping MS testing.")
            else:
                wri_vals = [int(wri_green), int(wri_red), int(wri_nir), int(wri_mir)]
                base = 1 if min(wri_vals) >= 1 else 0
                g_idx, r_idx, nir_idx, mir_idx = [v - base for v in wri_vals]

                stats_bands = sorted(set([g_idx, r_idx, nir_idx, mir_idx]))

                ms_dataset = MSPatchStatsDataset(
                    imgPath=topDir,
                    imgData=dataset_info,
                    labels=dataset_labels,
                    wri_bands=(g_idx, r_idx, nir_idx, mir_idx),
                    stats_bands=stats_bands,
                    train=False,
                    label_col=ms_label_col,
                    conf_col=ms_conf_col,
                )

                ms_loader = DataLoader(ms_dataset, batch_size=1, shuffle=False)

                feature_dim = ms_dataset.feature_dim
                if feature_dim == 0:
                    print("No MS features found; skipping MS testing.")
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

                    ms_checkpoint = torch.load(args.load_checkpoint_ms)
                    model_ms.load_state_dict(ms_checkpoint['state_dict'])

                    if args.cuda:
                        model_ms.cuda()

                    labels_ms = []
                    confs_ms = []

                    for batch_idx, (X_ms, _) in enumerate(ms_loader):
                        if batch_idx % 100 == 0:
                            print(f"Processing MS batch {batch_idx}")

                        if args.cuda:
                            X_ms = X_ms.to(device)

                        X_ms = X_ms.float()
                        Y_hat_ms = model_ms.forward(X_ms)
                        sm_ms = softmax(Y_hat_ms)
                        conf_ms = sm_ms.max()

                        if conf_ms > 0:
                            labels_ms.append(int(torch.argmax(Y_hat_ms)))
                            confs_ms.append(conf_ms.item())
                        else:
                            labels_ms.append(num_classes_ms)

                    split_info_ms = dataset[1]
                    split_info_ms[:, ms_label_col] = labels_ms
                    split_info_ms[:, ms_conf_col] = confs_ms
                    dataset[1] = split_info_ms

                    ms_checkpoint_str = os.path.basename(args.load_checkpoint_ms)
                    ms_output_dir = os.path.dirname(os.path.dirname(args.load_checkpoint_ms))
                    os.makedirs(os.path.join(ms_output_dir, "labels"), exist_ok=True)
                    np.save(os.path.join(ms_output_dir, "labels", "labeled_"+ms_checkpoint_str), dataset)

if args.netCDF:
    to_netCDF(dataset)

"""
