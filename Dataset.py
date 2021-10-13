import os
import utm
import rasterio as rio
import numpy as np
import math
from utils import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import random

#def load_split_images(img_mat, max, winSize):

class SplitImageDataset(Dataset):

    def __init__(self, imgPath, imgData, labels, transform=None, train=False):

        self.train = train
        imagePaths = getImgPaths(imgPath)
        imageLabels = labels
        imageData = imgData
        self.transform = transform

        # Extract all split images and store in dataframe (takes longer to initialize but saves loads on memory usage during training)
        dataArray = []

        for imgNum,imagePath in enumerate(imagePaths):

            # If training, and there are no labeled split images from tiff image, skip loading it
            if self.train and imageLabels[imageLabels[:,6] == imgNum].shape[0] == 0:
                continue

            img = rio.open(imagePath)
            imageMatrix = img.read(1)

            max = imageMatrix.max()

            winSize = imageData['winsize_pix']
            for row in imageLabels[imageLabels[:,6] == imgNum]:
                x,y = row[0:2].astype('int')
                splitImg_np = imageMatrix[x:x+winSize[0],y:y+winSize[1]]
                splitImg_np = scaleImage(splitImg_np, max)
                rowlist = list(row)
                rowlist.append(splitImg_np)
                dataArray.append(rowlist)

        self.dataFrame = pd.DataFrame(dataArray, columns=['x_pix','y_pix','x_utm','y_utm','label','conf','img_source','img_mat'])


    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):

        splitImg_np = self.dataFrame.iloc[idx,7]

        if self.transform:
            splitImg_np = self.transform(splitImg_np)

        splitImg_tensor = torch.from_numpy(splitImg_np)

        if self.train:
            label = int(self.dataFrame.iloc[idx,4])
            return (splitImg_tensor, label)

        else:
            return splitImg_tensor

class RandomRotateVario(object):

    def __init__(self):
        self.random = random.uniform(0,1)

    def __call__(self, vario):
        if self.random < 0.25:
            return np.concatenate((vario[0,:],vario[1,:],vario[2,:]))
        elif self.random < 0.5:
            return np.concatenate((vario[1,:],vario[0,:],vario[2,:]))
        elif self.random < 0.75:
            return np.concatenate((vario[0,:],vario[1,:],vario[3,:]))
        elif self.random < 1:
            return np.concatenate((vario[1,:],vario[0,:],vario[3,:]))

class DefaultRotateVario(object):

    def __call__(self, vario):
        return np.concatenate((vario[0,:],vario[1,:],vario[2,:]))

class DirectionalVario(object):

    def __init__(self, numLag):
        self.numLag = numLag

    def __call__(self, img):
        return directional_vario(img, self.numLag)

class RandomShift(object):

    def __call__(self, img):
        size = img.shape
        size_diff = abs(size[0]-size[1])
        rand = random.randint(0,size_diff-1)
        img = img[:,rand:]
        return img

class FlipHoriz(object):

    def __init__(self, threshold):
        self.random = random.uniform(0,1)
        self.threshold = threshold

    def __call__(self, sample):
        if self.random < self.threshold:
            sample = np.fliplr(sample)
        return sample

class FlipVert(object):

    def __init__(self, threshold):
        self.random = random.uniform(0,1)
        self.threshold = threshold

    def __call__(self, sample):
        if self.random < self.threshold:
            sample = np.flipud(sample)
        return sample

class AdjustContrast(object):

    def __init__(self, threshold):
        self.random = random.uniform(0,1)
        self.threshold = threshold

    def __call__(self, sample):
        if self.random < self.threshold:
            min=np.min(sample)        # result=144
            max=np.max(sample)        # result=216

            start = random.randint(0,np.min)
            stop = random.randint(np.max,255)

            # Make a LUT (Look-Up Table) to translate image values
            LUT=np.zeros(256,dtype=np.uint8)
            LUT[min:max+1]=np.linspace(start=start,stop=stop,num=(max-min)+1,endpoint=True,dtype=np.uint8)

            sample = LUT(sample)
            Image.fromarray(sample).save('result.png')

        return sample
