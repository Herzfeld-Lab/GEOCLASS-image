from torch.utils.data import DataLoader
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
import signal
from sklearn.utils.class_weight import compute_class_weight
import warnings

# Handle Ctrl-C event (manual stop training)
def signal_handler(sig, frame):
    save_losses()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def save_losses():
    np.save(output_dir+'/losses/'+checkpoint_str+"_train_losses",np.array(train_losses))
    np.save(output_dir+'/losses/'+checkpoint_str+"_valid_losses",np.array(valid_losses))
    plt.ylim([0,1])
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
hidden_layers = cfg['hidden_layers']

# Set dataset hyperparameters as specified by config file
topDir = cfg['img_path']
classEnum = cfg['class_enum']
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
    img_transforms_train = transforms.Compose([
        RandomShift(),
        DirectionalVario(model.num_lag),
        RandomRotateVario(),
    ])
    img_transforms_valid = transforms.Compose([
        DirectionalVario(model.num_lag),
        DefaultRotateVario(),
    ])
elif cfg['model'] == 'Resnet18':
    num_classes = cfg['num_classes']
    model = Resnet18.resnet18(pretrained=False, num_classes=num_classes)
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

print(model)

# Perform train/test split
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
dataset_coords = dataset[1]
if ddaBool:
    dataset_labeled = dataset_coords[dataset_coords[:,0] != -1]
else:
    dataset_labeled = dataset_coords[dataset_coords[:,4] != -1]
    
train_size = int(cfg['train_test_split'] * dataset_labeled.shape[0])

train_indeces = np.random.choice(range(dataset_labeled.shape[0]), train_size, replace=False)
test_indeces = list(set(range(dataset_labeled.shape[0])) - set(train_indeces))

train_coords = dataset_labeled[train_indeces, :]
test_coords = dataset_labeled[test_indeces, :]

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

# print('Training set size: \t%d images'%(len(train_dataset)))
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

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False
    )

weighted = True
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
    for batch_idx,(X,Y) in enumerate(train_loader):

        if batch_idx % int((len(train_dataset) / batch_size)/10) == 0:
            print('.', end='',flush=True)

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
    train_losses.append(sum_loss/batch_idx)

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
