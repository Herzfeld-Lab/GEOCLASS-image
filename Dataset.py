import os
import utm
import rasterio as rio
import numpy as np
import math
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PIL import ImageQt
from utils import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import random
import numpy

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
        #CST20240315print("image data", imageData)
        for imgNum,imagePath in enumerate(imagePaths):

            # If training, and there are no labeled split images from tiff image, skip loading it
            TimageLabels = list(zip(*imageLabels)) #CST20240322 this may fail or not work as expected now
            a=0
            if len(TimageLabels) == 7: #Training
                for i in range(0,len(TimageLabels[6])):
                    if TimageLabels[6][i]==imgNum:
                        a=1
                if self.train and a == 0:
                            continue

                img = rio.open(imagePath)
                imageMatrix = img.read(1)
                
                max = get_img_sigma(imageMatrix[::10,::10])
                winSize = imageData['winsize_pix']
            #CST 20240322
                for i in range(0,len(TimageLabels[6])):
                    if TimageLabels[6][i]==imgNum:
                        row = imageLabels[i]
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = list(row)
                        rowlist.append(splitImg_np)
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[4], "image source: ", rowlist[6])
                        else:
                            dataArray.append(rowlist)
                        #CST20240315print("data array", dataArray)
            elif len(TimageLabels) == 1: #testing
                    # If training, and there are no labeled split images from tiff image, skip loading it

                #CST 20240329
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
            return (splitImg_tensor, int(label))

        else:
            return splitImg_tensor
#For training by a folder of images
            
"""
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
        """

class CalipsoDataset(Dataset):

    def __init__(self, imgPath, imgData, labels, transform=None, train=False):

        self.train = train
        imagePath = imgPath
        imageLabels = labels
        imageData = imgData
        self.transform = transform
        TotDen = labels[:][5]
        density1 = TotDen[0]
        density2 = TotDen[1]
        density3 = TotDen[2]
        TotTab = labels[6]
        tab1 = TotTab[0]
        tab2 = TotTab[1]
        tab3 = TotTab[2]
        TotAsr = labels[7]
        asr1 = TotAsr[0]
        asr2 = TotAsr[1]
        asr3 = TotAsr[2]
        # Extract all split images and store in dataframe (takes longer to initialize but saves loads on memory usage during training)
        dataArray = []
        
        
        #CST20240315print("image data", imageData)
        if self.train:
                img = Image.open(imagePath)
                imageMatrix = np.array(img)
                max = get_img_sigma(imageMatrix[::10,::10])
                winSize = imageData['winsize_pix']
                for i in range(0,len(imageLabels[:])):
                        row = imageLabels[i]
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[y:y+winSize[1],x:x+winSize[0]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = list(row)
                        rowlist.append(splitImg_np)
                        rowlist.append(density1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(density2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(density3[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(tab1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(tab2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(tab3[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(asr1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(asr2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(asr3[y:y+winSize[1],x:x+winSize[0]])
                        #rowlist.append(self.labels_multiclass[i])
                        #rowlist.append(self.labels_binary[i])
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[2])
                        else:
                            dataArray.append(rowlist)
                        #CST20240315print("data array", dataArray)
        else: #testing
                    # If training, and there are no labeled split images from tiff image, skip loading it
                 

                img = Image.open(imagePath)
                imageMatrix = np.array(img)
                
                max = get_img_sigma(imageMatrix[::10,::10])
                winSize = imageData['winsize_pix']
                for i in range(0,len(imageLabels[:])):
                        row = imageLabels[i]
                        #print(row)
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[y:y+winSize[1],x:x+winSize[0]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = list(row)
                        rowlist.append(splitImg_np)
                        rowlist.append(density1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(density2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(density3[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(tab1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(tab2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(tab3[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(asr1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(asr2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(asr3[y:y+winSize[1],x:x+winSize[0]])
                        #rowlist.append(self.labels_multiclass[i])
                        #rowlist.append(self.labels_binary[i])
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[2])
                        else:
                            dataArray.append(rowlist)
                        

        self.dataFrame = pd.DataFrame(dataArray, columns=['x_pix','y_pix','label','conf','img_mat',
                                                          'density1','density2','density3','TAB1','TAB2','TAB3','ASR1','ASR2','ASR3'])

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        density1 = self.dataFrame.iloc[idx,5]
        density2 = self.dataFrame.iloc[idx,6]
        density3 = self.dataFrame.iloc[idx,7]
        tab1 = self.dataFrame.iloc[idx,8]
        tab2 = self.dataFrame.iloc[idx,9]
        tab3 = self.dataFrame.iloc[idx,10]
        asr1 = self.dataFrame.iloc[idx,11]
        asr2 = self.dataFrame.iloc[idx,12]
        asr3 = self.dataFrame.iloc[idx,13]

        #if self.transform:
            #splitImg_np = self.transform(splitImg_np)

        #density_tensor = torch.from_numpy([density1,density2,density3])
        #tab_tensor = torch.from_numpy([tab1,tab2,tab3])
        #asr_tensor = torch.from_numpy([asr1,asr2,asr3])
        train_tensor = torch.from_numpy([density1,density2,density3,tab1,tab2,tab3,asr1,asr2,asr3])

        if self.train:
            label = int(self.dataFrame.iloc[idx,2])
            return (train_tensor, int(label))

        else:
            return (train_tensor)
        



class CalipsoDataset2(Dataset): #gets density fields here instead of from dataset

    def __init__(self, imgPath, imgData, labels, den, tab, asr, tile_width, tile_height, transform=None, train=False):

        self.train = train
        imagePath = imgPath
        imageLabels = labels
        imageData = imgData
        self.transform = transform

        max = 0 #only will work if density conatins more than one entry
        max = 0
        for dens in den:
            for each_den in dens:
                if each_den > max:
                    max = each_den
                if each_den < 0:
                    each_den = 0
  
        
        img = den
        img_width, img_height = img.shape
        pad_height = (tile_height - img_height % tile_height) % tile_height
        pad_width = (tile_width - img_width % tile_width) % tile_width

    
        tab1 = tab[0]
        tab2 = tab[1]
        tab3 = tab[2]

        asr1 = asr[0]
        asr2 = asr[1]
        asr3 = asr[2]

        padded_tab1 = np.pad(
        tab1,
        ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
        mode='constant',
        constant_values=0  # Fill with 0s (can change)
        )   
        padded_tab2 = np.pad(
        tab2,
        ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
        mode='constant',
        constant_values=0  # Fill with 0s (can change)
        )   
        padded_tab3 = np.pad(
        tab3,
        ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
        mode='constant',
        constant_values=0  # Fill with 0s (can change)
        )   
        padded_asr1 = np.pad(
        asr1,
        ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
        mode='constant',
        constant_values=0  # Fill with 0s (can change)
        )   
        padded_asr2 = np.pad(
        asr2,
        ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
        mode='constant',
        constant_values=0  # Fill with 0s (can change)
        )   
        padded_asr3 = np.pad(
        asr3,
        ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
        mode='constant',
        constant_values=0  # Fill with 0s (can change)
        )   

        
        # Extract all split images and store in dataframe (takes longer to initialize but saves loads on memory usage during training)
        dataArray = []
        
        N = 15
        #CST20240315print("image data", imageData)
        if self.train:
                #img = Image.open(imagePath)
                imageMatrix = np.array(img)
                max = get_img_sigma(imageMatrix[::10,::10])
                winSize = imageData['winsize_pix']
                for i in range(0,len(imageLabels[:])):
                        row = imageLabels[i]
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[y:y+winSize[1],x:x+winSize[0]]
                        #splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = list(row[0:4])
                        rowlist.append(splitImg_np)
                        """
                        totTab1 = padded_tab1[y:y+winSize[1],x:x+winSize[0]]
                        totTab2 = padded_tab2[y:y+winSize[1],x:x+winSize[0]]
                        totTab3 = padded_tab3[y:y+winSize[1],x:x+winSize[0]]
                        totAsr1 = padded_asr1[y:y+winSize[1],x:x+winSize[0]]
                        totAsr2 = padded_asr2[y:y+winSize[1],x:x+winSize[0]]
                        totAsr3 = padded_asr3[y:y+winSize[1],x:x+winSize[0]]
                        avgTab1 = [pd.Series(totTab1[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totTab1[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgTab2 = [pd.Series(totTab2[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totTab2[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgTab3 = [pd.Series(totTab3[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totTab3[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgAsr1 = [pd.Series(totAsr1[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totAsr1[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgAsr2 = [pd.Series(totAsr2[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totAsr2[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgAsr3 = [pd.Series(totAsr3[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totAsr2[1]).rolling(window=N).mean().dropna().to_numpy()]
                        rowlist.append(avgTab1)
                        rowlist.append(avgTab2)
                        rowlist.append(avgTab3)
                        rowlist.append(avgAsr1)
                        rowlist.append(avgAsr2)
                        rowlist.append(avgAsr3)
                        """

                    
                        
                        totTab1 = padded_tab1[y:y+winSize[1],x:x+winSize[0]]
                        totTab2 = padded_tab2[y:y+winSize[1],x:x+winSize[0]]
                        totTab3 = padded_tab3[y:y+winSize[1],x:x+winSize[0]]
                        totAsr1 = padded_asr1[y:y+winSize[1],x:x+winSize[0]]
                        totAsr2 = padded_asr2[y:y+winSize[1],x:x+winSize[0]]
                        totAsr3 = padded_asr3[y:y+winSize[1],x:x+winSize[0]]
                        avgTab1 = [float(sum(totTab1[0])/len(totTab1[0])),float(sum(totTab1[1])/len(totTab1[1])), np.max(totTab1), np.min(totTab1)]
                        avgTab2 = [float(sum(totTab2[0])/len(totTab2[0])),float(sum(totTab2[1])/len(totTab2[1])), np.max(totTab1), np.min(totTab1)]
                        avgTab3 = [float(sum(totTab3[0])/len(totTab3[0])),float(sum(totTab3[1])/len(totTab3[1])), np.max(totTab1), np.min(totTab1)]
                        avgAsr1 = [float(sum(totAsr1[0])/len(totAsr1[0])),float(sum(totAsr1[1])/len(totAsr1[1])), np.max(totTab1), np.min(totTab1)]
                        avgAsr2 = [float(sum(totAsr2[0])/len(totAsr2[0])),float(sum(totAsr2[1])/len(totAsr2[1])), np.max(totTab1), np.min(totTab1)]
                        avgAsr3 = [float(sum(totAsr3[0])/len(totAsr3[0])),float(sum(totAsr3[1])/len(totAsr3[1])), np.max(totTab1), np.min(totTab1)]
                        height = [y,0,0,0]
                        """
                        avgTab1 = [float(sum(totTab1[0])/len(totTab1[0])),float(sum(totTab1[1])/len(totTab1[1])), np.std(totTab1[0]), np.std(totTab1[1]), np.max(totTab1)-np.min(totTab1)]
                        avgTab2 = [float(sum(totTab2[0])/len(totTab2[0])),float(sum(totTab2[1])/len(totTab2[1])), np.std(totTab2[0]), np.std(totTab2[1]), np.max(totTab1)-np.min(totTab1)]
                        avgTab3 = [float(sum(totTab3[0])/len(totTab3[0])),float(sum(totTab3[1])/len(totTab3[1])), np.std(totTab3[0]), np.std(totTab3[1]), np.max(totTab1)-np.min(totTab1)]
                        avgAsr1 = [float(sum(totAsr1[0])/len(totAsr1[0])),float(sum(totAsr1[1])/len(totAsr1[1])), np.std(totAsr1[0]), np.std(totAsr1[1]), np.max(totTab1)-np.min(totTab1)]
                        avgAsr2 = [float(sum(totAsr2[0])/len(totAsr2[0])),float(sum(totAsr2[1])/len(totAsr2[1])), np.std(totAsr2[0]), np.std(totAsr2[1]), np.max(totTab1)-np.min(totTab1)]
                        avgAsr3 = [float(sum(totAsr3[0])/len(totAsr3[0])),float(sum(totAsr3[1])/len(totAsr3[1])), np.std(totAsr3[0]), np.std(totAsr3[1]), np.max(totTab1)-np.min(totTab1)]
                        """
                        
                        rowlist.append(avgTab1)
                        rowlist.append(avgTab2)
                        rowlist.append(avgTab3)
                        rowlist.append(avgAsr1)
                        rowlist.append(avgAsr2)
                        rowlist.append(avgAsr3)
                        rowlist.append(height)
                        
                        """
                        rowlist.append(padded_tab1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_tab2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_tab3[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_asr1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_asr2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_asr3[y:y+winSize[1],x:x+winSize[0]])
                        """
                        #rowlist.append(self.labels_multiclass[i])
                        #rowlist.append(self.labels_binary[i])
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[2])
                        else:
                            dataArray.append(rowlist)
                        #CST20240315print("data array", dataArray)
        else: #testing
                    # If training, and there are no labeled split images from tiff image, skip loading it
                 
                imageMatrix = np.array(img)
                
                max = get_img_sigma(imageMatrix[::10,::10])
                winSize = imageData['winsize_pix']
                for i in range(0,len(imageLabels[:])):
                        row = imageLabels[i]
                        #print(row)
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[y:y+winSize[1],x:x+winSize[0]]
                        #splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = list(row[0:4])
                        rowlist.append(splitImg_np)

                        """
                        totTab1 = padded_tab1[y:y+winSize[1],x:x+winSize[0]]
                        totTab2 = padded_tab2[y:y+winSize[1],x:x+winSize[0]]
                        totTab3 = padded_tab3[y:y+winSize[1],x:x+winSize[0]]
                        totAsr1 = padded_asr1[y:y+winSize[1],x:x+winSize[0]]
                        totAsr2 = padded_asr2[y:y+winSize[1],x:x+winSize[0]]
                        totAsr3 = padded_asr3[y:y+winSize[1],x:x+winSize[0]]
                        
                        avgTab1 = [pd.Series(totTab1[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totTab1[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgTab2 = [pd.Series(totTab2[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totTab2[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgTab3 = [pd.Series(totTab3[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totTab3[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgAsr1 = [pd.Series(totAsr1[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totAsr1[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgAsr2 = [pd.Series(totAsr2[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totAsr2[1]).rolling(window=N).mean().dropna().to_numpy()]
                        avgAsr3 = [pd.Series(totAsr3[0]).rolling(window=N).mean().dropna().to_numpy(), pd.Series(totAsr2[1]).rolling(window=N).mean().dropna().to_numpy()]
                        rowlist.append(avgTab1)
                        rowlist.append(avgTab2)
                        rowlist.append(avgTab3)
                        rowlist.append(avgAsr1)
                        rowlist.append(avgAsr2)
                        rowlist.append(avgAsr3)

                        """
                        totTab1 = padded_tab1[y:y+winSize[1],x:x+winSize[0]]
                        totTab2 = padded_tab2[y:y+winSize[1],x:x+winSize[0]]
                        totTab3 = padded_tab3[y:y+winSize[1],x:x+winSize[0]]
                        totAsr1 = padded_asr1[y:y+winSize[1],x:x+winSize[0]]
                        totAsr2 = padded_asr2[y:y+winSize[1],x:x+winSize[0]]
                        totAsr3 = padded_asr3[y:y+winSize[1],x:x+winSize[0]]
                        avgTab1 = [float(sum(totTab1[0])/len(totTab1[0])),float(sum(totTab1[1])/len(totTab1[1])), np.max(totTab1), np.min(totTab1)]
                        avgTab2 = [float(sum(totTab2[0])/len(totTab2[0])),float(sum(totTab2[1])/len(totTab2[1])), np.max(totTab1), np.min(totTab1)]
                        avgTab3 = [float(sum(totTab3[0])/len(totTab3[0])),float(sum(totTab3[1])/len(totTab3[1])), np.max(totTab1), np.min(totTab1)]
                        avgAsr1 = [float(sum(totAsr1[0])/len(totAsr1[0])),float(sum(totAsr1[1])/len(totAsr1[1])), np.max(totTab1), np.min(totTab1)]
                        avgAsr2 = [float(sum(totAsr2[0])/len(totAsr2[0])),float(sum(totAsr2[1])/len(totAsr2[1])), np.max(totTab1), np.min(totTab1)]
                        avgAsr3 = [float(sum(totAsr3[0])/len(totAsr3[0])),float(sum(totAsr3[1])/len(totAsr3[1])), np.max(totTab1), np.min(totTab1)]
                        height = [y,0,0,0]
                        """
                        avgTab1 = [float(sum(totTab1[0])/len(totTab1[0])),float(sum(totTab1[1])/len(totTab1[1])), np.std(totTab1[0]), np.std(totTab1[1]), np.max(totTab1)-np.min(totTab1)]
                        avgTab2 = [float(sum(totTab2[0])/len(totTab2[0])),float(sum(totTab2[1])/len(totTab2[1])), np.std(totTab2[0]), np.std(totTab2[1]), np.max(totTab1)-np.min(totTab1)]
                        avgTab3 = [float(sum(totTab3[0])/len(totTab3[0])),float(sum(totTab3[1])/len(totTab3[1])), np.std(totTab3[0]), np.std(totTab3[1]), np.max(totTab1)-np.min(totTab1)]
                        avgAsr1 = [float(sum(totAsr1[0])/len(totAsr1[0])),float(sum(totAsr1[1])/len(totAsr1[1])), np.std(totAsr1[0]), np.std(totAsr1[1]), np.max(totTab1)-np.min(totTab1)]
                        avgAsr2 = [float(sum(totAsr2[0])/len(totAsr2[0])),float(sum(totAsr2[1])/len(totAsr2[1])), np.std(totAsr2[0]), np.std(totAsr2[1]), np.max(totTab1)-np.min(totTab1)]
                        avgAsr3 = [float(sum(totAsr3[0])/len(totAsr3[0])),float(sum(totAsr3[1])/len(totAsr3[1])), np.std(totAsr3[0]), np.std(totAsr3[1]), np.max(totTab1)-np.min(totTab1)]
                        """
                        
                        rowlist.append(avgTab1)
                        rowlist.append(avgTab2)
                        rowlist.append(avgTab3)
                        rowlist.append(avgAsr1)
                        rowlist.append(avgAsr2)
                        rowlist.append(avgAsr3)
                        rowlist.append(height)

                        """
                        rowlist.append(padded_tab1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_tab2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_tab3[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_asr1[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_asr2[y:y+winSize[1],x:x+winSize[0]])
                        rowlist.append(padded_asr3[y:y+winSize[1],x:x+winSize[0]])
                        """
                        #rowlist.append(self.labels_multiclass[i])
                        #rowlist.append(self.labels_binary[i])
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[2])
                        else:
                            dataArray.append(rowlist)
                        

        self.dataFrame = pd.DataFrame(dataArray, columns=['x_pix','y_pix','label','conf','img_mat', 'TAB1','TAB2','TAB3','ASR1','ASR2','ASR3', 'y'])

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):

        tab1 = self.dataFrame.iloc[idx,5]
        tab2 = self.dataFrame.iloc[idx,6]
        tab3 = self.dataFrame.iloc[idx,7]
        asr1 = self.dataFrame.iloc[idx,8]
        asr2 = self.dataFrame.iloc[idx,9]
        asr3 = self.dataFrame.iloc[idx,10]
        y = self.dataFrame.iloc[idx,11]
        np_array = np.array([tab1,tab2,tab3,asr1,asr2,asr3,y])

        #if self.transform:
            #splitImg_np = self.transform(splitImg_np)

        #density_tensor = torch.from_numpy([density1,density2,density3])
        #tab_tensor = torch.from_numpy([tab1,tab2,tab3])
        #asr_tensor = torch.from_numpy([asr1,asr2,asr3])
        train_tensor = torch.from_numpy(np_array)

        if self.train:
            label = int(self.dataFrame.iloc[idx,2])
            return (train_tensor, int(label))

        else:
            return (train_tensor)





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
        imSize = img.shape
        if (imSize[0] == 201 and imSize[1] == 268) or (imSize[0] == 268 and imSize[1] == 201):
            return silas_directional_vario(img, self.numLag)
        else:
            print("Use an image size of (201,268) for best results")
            return fast_directional_vario(img, self.numLag)
        

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

class DDAiceDataset(Dataset):

    def __init__(self, dataPath, dataInfo, dataLabeled, transform=None, train=False):

        self.train = train
        self.transform = transform
        ddaGroundEstPath = dataPath[0] # path to ground estimate
        datasetInfo = dataInfo
        variograms = dataLabeled

        # Work on configuring pandas data frame - numpy easier right now for 48 col array
        # cols = ['lon','lat','utm_e','utm_n','dist','delta_time','pond','p1','p2','mindist','hdiff','nugget','photon_density','variogram','label']
        # labels = dataLabeled[:,cutoff]
        # variograms = dataLabeled[:,0:cutoff]
        # dataDict = {'label': labels, 'variogram': variograms}

        # data format: [label, conf, ge varios (nres-1) columns, wp varios (nres-1) columns]
        self.dataFrame = variograms
        # self.dataFrame = dataDict

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self,idx):
        vario = self.dataFrame[idx,2:]
        # vario = self.dataFrame['variogram'][idx]

        vario_tensor = torch.from_numpy(vario)

        if self.train:
            label = int(self.dataFrame[idx,0])
            # label = int(self.dataFrame['label'][idx])
            return (vario_tensor, label)
        else:
            return vario_tensor

    def get_labels(self):
        return self.dataFrame[:,0]