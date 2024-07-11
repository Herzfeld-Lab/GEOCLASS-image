import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from utils import *
import cv2
import rasterio as rio

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
    label_map = {}  # To store the mapping from folder names to label indices
    label_counter = 0

    for label_name in os.listdir(image_folder):
        label_path = os.path.join(image_folder, label_name)
        if os.path.isdir(label_path):
            if label_name not in label_map:
                label_map[label_name] = label_counter
                label_counter += 1
            label_index = label_map[label_name]
            for img_name in os.listdir(label_path):
                if img_name.endswith(('png', 'tiff', 'tif')):
                    img_path = os.path.join(label_path, img_name)
                    image_paths.append(img_path)
                    labels.append(label_index)
                    # Load image and convert to numpy array
                    image = Image.open(img_path).convert('RGB')
                    image_np = np.array(image)
                    variogram = silas_directional_vario(image_np)
                    variograms.append(variogram)
    return image_paths, np.array(variograms), labels, label_map

#Labels are just in order, not the actually class so this needs to change in final implentation
# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_paths, variogram_data, labels, transform=None):
        self.image_paths = image_paths
        self.variogram_data = variogram_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = rio.open(image_path)
        except Exception as e:
            print(f"Error opening image at {image_path}: {str(e)}")
            raise e
        
        variogram = self.variogram_data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, variogram, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet18 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Use grayscale mean and std
])



resnet = models.resnet18(pretrained=True)

# Modify the first convolutional layer to accept single-channel input
num_ftrs = resnet.fc.in_features
original_conv1 = resnet.conv1
resnet.conv1 = nn.Conv2d(1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size, 
                         stride=original_conv1.stride, padding=original_conv1.padding, bias=original_conv1.bias)

# Initialize the new layer's weights by averaging the original layer's weights
with torch.no_grad():
    resnet.conv1.weight = nn.Parameter(torch.mean(original_conv1.weight, dim=1, keepdim=True))


# Define the custom neural network
class CombinedNN(nn.Module):
    def __init__(self, resnet, variogram_size, num_classes):
        super(CombinedNN, self).__init__()
        self.resnet = resnet
        
        # Determine the output dimension of ResNet's final layer
        resnet_output_dim = 512  # This should match the output dimension of ResNet-18
        
        # Define fully connected layers
        self.fc1 = nn.Linear(variogram_size + 1000, 512) #The 1000 is the size of the image features but I'm not sure if this is always 1000 for resnet18
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, image, variogram):
        # Extract image features using ResNet
        image_features = self.resnet(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten image features
        
        # Flatten and process variogram data
        variogram = variogram.float()
        variogram = variogram.view(variogram.size(0), -1)
        # Concatenate image features and variogram features
        combined_features = torch.cat((image_features, variogram), dim=1)
        # Forward pass through fully connected layers
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x
"""
image_folder = 'Classification'
num_classes = 11
image_paths, variogram_data, labels, label_map = collect_image_paths_and_labels(image_folder)
variogram_size = variogram_data[0][0].shape[0]*variogram_data[0].shape[0]
train_size = int(0.8 * len(image_paths))
train_indeces = np.random.choice(range(np.array(len(image_paths))), train_size, replace=False)
test_indeces = np.setdiff1d(range(np.array(len(image_paths))), train_indeces)
#CST20240322 Creating loops so train and test coords aren't 1D
train_imgs = []
test_imgs = []
train_var = []
test_var = []
train_labels = []
test_labels = []

for i in train_indeces:
    train_imgs.append(image_paths[i])
    train_var.append(image_paths[i])
    train_labels.append(image_paths[i])
for i in test_indeces:
    test_imgs.append(image_paths[i])
    test_var.append(image_paths[i])
    test_labels.append(image_paths[i])
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Identity()  # Remove the final fully connected layer
model = CombinedNN(resnet18, variogram_size, num_classes)
train_dataset = CustomDataset(train_imgs, train_var, train_labels, transform)
"""