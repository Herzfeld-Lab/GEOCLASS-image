import os
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
from Models import *
from Models import VarioMLP
from Models import Resnet18


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
                    variogram = get_varios(image_np)
                    variograms.append(variogram)
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
    def __init__(self, imgPath, imgData, labels, train):
        self.train = train
        imagePaths = getImgPaths(imgPath)
        imageLabels = labels
        imageData = imgData
        # Extract all split images and store in dataframe
        dataArray = []
        self.varios = []  # Initialize self.varios as an instance attribute
        
        for imgNum, imagePath in enumerate(imagePaths):
            print("\nCalculating Variograms for Image", imgNum)
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
                    if i % 1000 == 0:
                        print('.', end='', flush=True)
                    if TimageLabels[6][i] == imgNum:
                        row = imageLabels[i]
                        x, y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[x:x + winSize[0], y:y + winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max_sigma)
                        variogram = get_varios(splitImg_np)
                        self.varios.append(variogram)
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
                        variogram = get_varios(splitImg_np)
                        self.varios.append(variogram)
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
        variograms = self.varios[idx]/100 #decreases effect on network

        splitImg_tensor = torch.from_numpy(splitImg_np)
        vario_tensor = torch.from_numpy(variograms)
        if self.train:
            label = int(self.dataFrame.iloc[idx,4])
            return (splitImg_tensor, vario_tensor, label)
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
        else:
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
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
            variogram = self.variogram_data[idx]/100 #decreases effect on network
            return IMGnp, variogram, int(label)
        else:
            return IMGnp, int(label)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
])

class CombinedModel(nn.Module):
    def __init__(self, vario_mlp, resnet18, num_classes):
        super(CombinedModel, self).__init__()
        self.vario_mlp = vario_mlp
        self.resnet18 = resnet18
        
        # Freeze the weights of VarioMLP and ResNet18
        for param in self.vario_mlp.parameters():
            param.requires_grad = False
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        # Final fully connected layer after combining features
        combined_output_size = num_classes * 2
        self.fc = nn.Linear(combined_output_size, num_classes)

    def forward(self, image_data, variogram_data):
        vario_out = self.vario_mlp(variogram_data)
        resnet_out = self.resnet18(image_data)
        
        # Combine the outputs (e.g., concatenation)
        combined_out = torch.cat((vario_out, resnet_out), dim=1)
        
        # Final output
        final_out = self.fc(combined_out)
        
        return final_out