import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset as dataset
from PIL import Image
import numpy as np
from utils import *
import cv2
import rasterio as rio
import numpy
import yaml
import argparse
import torch.nn.functional as F
from datetime import datetime
from Models import *
from Models import VarioMLP
from Models import Resnet18
from scipy.optimize import linprog
import sklearn
from sklearn.neighbors import KernelDensity


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--netCDF", action="store_true")
args = parser.parse_args()
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load image with alpha channel if present
        if img is None:
            print(f"Error: Could not read image at {path}")
        else:
            images.append(img)
    return images

# Function to collect image paths, variogram data, and labels
def collect_image_paths_and_labels(image_folder):
    image_paths = []
    labels = []
    variograms = []

    for label_name in os.listdir(image_folder):
        label_path = os.path.join(image_folder, label_name)
        if os.path.isdir(label_path):
            label_index = int(label_name)
            for img_name in os.listdir(label_path):
                if img_name.endswith(('png', 'tiff', 'tif')):
                    img_path = os.path.join(label_path, img_name)
                    image_paths.append(img_path)
                    labels.append(label_index)
                    # Load image and convert to numpy array
                    image = Image.open(img_path).convert('RGB')
                    image_np = np.array(image)
                    vario = get_varios(image_np)
                    #Random rotate of varios
                    rand = random.uniform(0,1)

                    if rand < 0.25:
                        vario=np.concatenate((vario[0,:],vario[1,:],vario[2,:]))
                    elif rand < 0.5:
                        vario=np.concatenate((vario[1,:],vario[0,:],vario[2,:]))
                    elif rand < 0.75:
                        vario=np.concatenate((vario[0,:],vario[1,:],vario[3,:]))
                    elif rand < 1:
                        vario=np.concatenate((vario[1,:],vario[0,:],vario[3,:]))
                    variograms.append(vario)
    return image_paths, np.array(variograms), labels

def get_varios(img):
    numLag = cfg['vario_num_lag']
    imSize = img.shape
    if (imSize[0] == 201 and imSize[1] == 268) or (imSize[0] == 268 and imSize[1] == 201):
        return silas_directional_vario(img, numLag)
    else:
        print("Use an image size of (201,268) for best results")
        return fast_directional_vario(img, numLag)
#Labels are just in order, not the actual class so this needs to change in final implentation
#Testing Dataset

class TestDataset(dataset):
    def __init__(self, imgPath, imgData, labels, train, transform = None):
        self.train = train
        imagePaths = getImgPaths(imgPath)
        imageLabels = labels
        imageData = imgData
        self.transform = transform
        # Extract all split images and store in dataframe
        dataArray = []
        self.varios = []  # Initialize self.varios as an instance attribute
        
        for imgNum, imagePath in enumerate(imagePaths):
            TimageLabels = list(zip(*imageLabels))
            a=0
            if len(TimageLabels) == 7:  # Training
                for i in range(0,len(TimageLabels[6])):
                    if TimageLabels[6][i]==imgNum:
                        a=1
                if self.train and a == 0:
                            continue
                img = rio.open(imagePath)
                imageMatrix = img.read(1)
                
                max_sigma = get_img_sigma(imageMatrix[::10, ::10])
                winSize = imageData['winsize_pix']
                
                for i in range(len(TimageLabels[6])):
                    if TimageLabels[6][i] == imgNum:
                        row = imageLabels[i]
                        x, y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[x:x + winSize[0], y:y + winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max_sigma)
                        rowlist = list(row)
                        rowlist.append(splitImg_np)
                        if splitImg_np.shape[0] == 0 or splitImg_np.shape[1] == 0:
                            print("Error with an image: ", i, "class: ", rowlist[4], "image source: ", rowlist[6])
                        else:
                            dataArray.append(rowlist)
            elif len(TimageLabels) == 1: #testing
                    # If training, and there are no labeled split images from tiff image, skip loading it
                for i in range(0,len(TimageLabels[0])):
                    if TimageLabels[0][i][6]==imgNum:
                        a=1
                if self.train and a == 0:
                            continue
                    

                img = rio.open(imagePath)
                imageMatrix = img.read(1)
                
                max = get_img_sigma(imageMatrix[::10,::10])
                winSize = imageData['winsize_pix']
                #CST 20240329
                for i in range(0,len(TimageLabels[0])):
                    if TimageLabels[0][i][6] == imgNum:
                        row = imageLabels[i][0]
                        #print(row)
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = list(row)
                        rowlist.append(splitImg_np)
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[4], "image source: ", rowlist[6])
                        else:
                            dataArray.append(rowlist)
            else:
                print("Error with training or testing data")
                
        self.dataFrame = pd.DataFrame(dataArray, columns=['x_pix', 'y_pix', 'x_utm', 'y_utm', 'label', 'conf', 'img_source', 'img_mat'])

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        splitImg_np = self.dataFrame.iloc[idx, 7]
        if self.transform:
            vario_tensor = self.transform(splitImg_np)/10
        else:
            vario_tensor = get_varios(splitImg_np)/10
        splitImg_tensor = torch.from_numpy(splitImg_np)

        if self.train:
            label = int(self.dataFrame.iloc[idx,4])
            return (splitImg_tensor, vario_tensor, int(label))
        else:
            return splitImg_tensor, vario_tensor


# Custom Dataset Class
class FromFolderDataset(dataset):
    def __init__(self, model, image_paths, variogram_data, labels, transform=None):
        self.model = model
        if self.model == 'VarioNet':
            self.image_paths = image_paths
            self.variogram_data = variogram_data
            self.labels = labels
            self.transform = transform
        elif self.model == 'VarioMLP':
            self.variogram_data = variogram_data
            self.labels = labels
            self.transform = transform
        else:
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            

    def __len__(self):
        if self.model == 'VarioMLP':
            return len(self.labels)
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        if self.model == 'VarioMLP':
            variogram = self.variogram_data[idx]
            return variogram, int(label)
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image at {image_path}: {str(e)}")
            raise e
        IMGnp = numpy.array(image)
        
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.model == 'VarioNet':
            variogram = self.variogram_data[idx]/10#should decreases effect on network
            return IMGnp, variogram, int(label)
        else:
            return IMGnp, int(label)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
])

#VarioNet 

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, downsample=False):
        super(MLPBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)  # Expanding the feature size
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)  # Keep output size same as out_features
        
        self.downsample = None
        if downsample:
            self.downsample = nn.Linear(in_features, out_features)  # Adjust dimensions

    def forward(self, x):
        identity = x  # Store original input
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)  # Adjust identity mapping if needed

        out += identity  # Residual connection
        out = self.relu(out)
        
        return out
"""
class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        residual = x  # Store original input
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual  # Add input to output
        return F.relu(out)

class MLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, downsample=False):
        super(MLPBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, in_features)  # Ensure residual connection works
        
        self.downsample = None
        if downsample:
            self.downsample = nn.Linear(in_features, hidden_features)  # Adjust dimensions if needed

    def forward(self, x):
        identity = x  # Store the original input
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)  # Apply downsampling if required

        out += identity  # Residual connection
        out = self.relu(out)  # Final activation
        
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(BottleneckBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)  # Reduce dim
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)  # Expand back

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out)
"""


class CombinedModel(nn.Module):

    
    def __init__(self, variomlp, resnet, num_classes,  num_blocks = 1,  a=0.5, b=0.5, adaptive = False):
        super(CombinedModel, self).__init__()
        self.b_values = [] 
        self.vario_mlp = variomlp
        self.resnet18 = resnet
        self.adaptive = adaptive
        # Freeze the weights of VarioMLP and ResNet18
        
        for param in self.vario_mlp.parameters():
            param.requires_grad = False
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        
        #dummy_vario = torch.randn(1, self.vario_mlp.input_size)  # Adjust shape based on vario input
        #dummy_image = torch.randn(64, 1, 7, 7)  # Adjust shape based on ResNet input (default: 3x224x224)

        #vario__dim = self.vario_mlp.get_intermediate_features(dummy_vario).numel()

        self.alpha = nn.Parameter(torch.tensor(a))
        self.beta = nn.Parameter(torch.tensor(b))

        layers = []
        in_dim = num_classes
        
        for i in range(num_blocks):
            out_dim = 64 * (i+1) # Double the hidden size
            layers.append(MLPBlock(in_dim, out_dim, downsample=True))
            in_dim = out_dim  # Update input dimension for the next block
            #if i == 1:
                #layers.append(BottleneckBlock(in_dim, (in_dim//2)))
        #self.fc_var = nn.Linear(vario__dim, 512)
        #self.fc_res = nn.Linear(512, 512)
        #self.fc1 = nn.Linear(512, 20)
        self.blocks = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_dim, num_classes)  # Final output layer
        #self.fc_res = nn.Linear(num_classes, num_classes)

    def forward(self, image_data, variogram_data):
        #vario_features = self.vario_mlp.get_intermediate_features(variogram_data)
        #resnet_features = self.resnet18.get_intermediate_features(image_data)

        #vario_out = F.relu(self.fc_var(vario_features))
        #resnet_out = F.relu(self.fc_res(resnet_features))
        softmax = torch.nn.Softmax(dim=1)
        vario_out = self.vario_mlp(variogram_data)
        resnet_out = self.resnet18(image_data)
        if self.adaptive:
            
            sm_res = softmax(resnet_out)
            sm_var = softmax(vario_out)
            #b = (sm_res.max()+(1-sm_var.max()))/2
            b = sm_res.max()
            self.b_values.append(b.item())
            a = 1-b
            self.beta = nn.Parameter(b.clone().detach().requires_grad_(True))
            self.alpha = nn.Parameter(a.clone().detach().requires_grad_(True))
    
        x = self.alpha * softmax(vario_out) + self.beta * softmax(resnet_out)
        #x = torch.cat((vario_out,resnet_out), dim=1)
        #x = self.fc1(x)
        x = self.blocks(x)
        x = self.fc_out(x)

        return x



    def plot_beta(self, output_dir, conf):
        """
        if isinstance(self.b_values, list):  
            b_values = np.array([x.detach().cpu().item() if isinstance(x, torch.Tensor) else x for x in self.b_values])
        elif isinstance(self.b_values, torch.Tensor):  
            b_values = self.b_values.detach().cpu().numpy()  # Move tensor to CPU
        else:
            raise TypeError(f"Unsupported type for b_values: {type(self.b_values)}")
        if isinstance(conf, list):  
            conf = np.array([x.detach().cpu().item() if isinstance(x, torch.Tensor) else x for x in conf])
        elif isinstance(conf, torch.Tensor):  
            conf = conf.detach().cpu().numpy()  # Move tensor to CPU
        else:
            raise TypeError(f"Unsupported type for b_values: {type(conf)}")
        """


        if not os.path.exists(output_dir+'/plots'): os.mkdir(output_dir+'/plots')
        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y_%H:%M")
        #unique_values, counts = np.unique(self.b_values, return_counts=True)
        data = np.vstack([self.b_values, conf]).T

        # Fit the KDE model
        kde = KernelDensity(bandwidth=0.1)
        kde.fit(data)

        # Evaluate the density model on the data points
        density = np.exp(kde.score_samples(data))

        normalized_density = (density - np.min(density)) / (np.max(density) - np.min(density))

        # Plot the scatter plot with density-based coloring
        plt.figure(figsize=(8, 6))
        plt.scatter(self.b_values, conf, c=normalized_density, s=50, cmap='viridis', edgecolors='k', alpha=0.7)
        plt.colorbar(label='Density')
        # Labels and title
        plt.xlabel("Beta")
        plt.ylabel("Confidence")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title("Effect of Beta on the Confidence of VarioNet")
        plt.savefig(output_dir+'/plots/'+date_str+'_beta_conf')

        # Show plot



"""
#Old code
class CombinedModel(nn.Module):
    def __init__(self, vario_mlp, resnet18, num_classes, a=0.5, b=0.5):
        super(CombinedModel, self).__init__()
        self.vario_mlp = vario_mlp
        self.resnet18 = resnet18
        
        # Freeze the weights of VarioMLP and ResNet18
        for param in self.vario_mlp.parameters():
            param.requires_grad = False
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        # Final fully connected layer after combining features
        combined_output_size = num_classes
        self.fc1 = nn.Linear(combined_output_size, 256)  # Bottleneck 1
        self.residual_fc1 = nn.Linear(combined_output_size, 256)
        #self.fc2 = nn.Linear(256, num_classes) # Bottleneck 2
        #self.residual_fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(256, num_classes) 


        self.alpha = nn.Parameter(torch.tensor(a))
        self.beta = nn.Parameter(torch.tensor(b))

    def forward(self, image_data, variogram_data):
        vario_out = self.vario_mlp(variogram_data)
        resnet_out = self.resnet18(image_data)
        
        # Weighted sum fusion
        combined_out = self.alpha * vario_out + self.beta * resnet_out
        
        # Bottleneck layer
        residual = self.residual_fc1(combined_out)  # Store original combined features
        combined_out = F.relu(self.fc1(combined_out))
        combined_out = combined_out + residual  # Add back original combined features
        #residual = self.residual_fc2(combined_out)
        #combined_out = F.relu(self.fc2(combined_out))
        #combined_out = combined_out + residual
        final_out = self.fc3(combined_out)
        
        return final_out
"""


    