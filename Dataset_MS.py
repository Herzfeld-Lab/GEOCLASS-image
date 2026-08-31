import os
import utm
import rasterio as rio
import numpy as np
import math
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PIL import ImageQt
from utils_MS import *
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

class SplitImageDatasetPAN(Dataset):

    def __init__(self, imgPath, imgData, labels, transform=None, train=False):

        self.train = train
        imagePaths = getImgPathsPan(imgPath)
        imageLabels = labels
        imageData = imgData
        self.transform = transform
        # Extract all split images and store in dataframe (takes longer to initialize but saves loads on memory usage during training)
        dataArray = []
        #CST20240315print("image data", imageData)
        #[pan_x, pan_y, ms_x, ms_y, utm_x, utm_y, pan_label, pan_conf, ms_label, ms_conf, img_num]
        def to_pan_rowlist(row, split_img):
            r = np.array(row)
            rowlist = [r[0], r[1], r[4], r[5], r[6], r[7], r[10]]
            rowlist.append(split_img)
            return rowlist

        for imgNum,imagePath in enumerate(imagePaths):

            # If training, and there are no labeled split images from tiff image, skip loading it
            TimageLabels = list(zip(*imageLabels)) #CST20240322 this may fail or not work as expected now
            a=0
       
            if len(TimageLabels) == 11: #Training
                for i in range(0,len(TimageLabels[10])):
                    if TimageLabels[10][i]==imgNum:
                        a=1
                if self.train and a == 0:
                            continue

                img = rio.open(imagePath)
                imageMatrix = img.read(1)
                
                max = get_img_sigma(imageMatrix[::10,::10])
                winSize = imageData['winsize_pix']
            #CST 20240322
                for i in range(0,len(TimageLabels[10])):
                    if TimageLabels[10][i]==imgNum:
                        row = imageLabels[i]
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = to_pan_rowlist(row, splitImg_np)
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[4], "image source: ", rowlist[6])
                        else:
                            dataArray.append(rowlist)
                        #CST20240315print("data array", dataArray)
            elif len(TimageLabels) == 1: #testing
                    # If training, and there are no labeled split images from tiff image, skip loading it

                #CST 20240329
                for i in range(0,len(TimageLabels[0])):
                    if TimageLabels[0][i][10]==imgNum:
                        a=1
                if self.train and a == 0:
                            continue
                    

                img = rio.open(imagePath)
                imageMatrix = img.read(1)
                
                max = get_img_sigma(imageMatrix[::10,::10])
                winSize = imageData['winsize_pix']
                #CST 20240329
                for i in range(0,len(TimageLabels[0])):
                    if TimageLabels[0][i][10] == imgNum:
                        row = imageLabels[i][0]
                        #print(row)
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = to_pan_rowlist(row, splitImg_np)
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

class SplitImageDatasetMS(Dataset):

    def __init__(self, imgPath, imgData, labels, transform=None, train=False):
        """
        Multispectral split-image dataset.
        Uses ms_x/ms_y from the label rows so MS and pan cover the same UTM region.
        Expected labels (per row):
          [pan_x, pan_y, ms_x, ms_y, utm_x, utm_y, pan_label, pan_conf, ms_label, ms_conf, img_num]
        """
        self.train = train
        imagePaths = getImgPathsMS(imgPath)
        imageLabels = labels
        imageData = imgData
        self.transform = transform
        # If the saved dataset info includes the original filename ordering, use it
        filenames = None
        try:
            filenames = imageData.get('filename', None)
        except Exception:
            filenames = None

        dataArray = []
        indices_list = []
        def to_ms_rowlist(row, split_img):
            r = np.array(row)
            # Keep only MS-relevant fields: ms_x, ms_y, utm_x, utm_y, ms_label, conf, img_source
            rowlist = [r[2], r[3], r[4], r[5], r[8], r[9], r[10]]
            rowlist.append(split_img)
            return rowlist
        for imgNum, imagePath in enumerate(imagePaths):
            # If training, and there are no labeled split images from tiff image, skip loading it
            TimageLabels = list(zip(*imageLabels)) #CST20240322 this may fail or not work as expected now
            if len(TimageLabels) == 11: #Training
                file_idx = imgNum
                if filenames is not None:
                    try:
                        file_idx = filenames.index(imagePath)
                    except ValueError:
                        bas = os.path.basename(imagePath)
                        found = -1
                        for k, fp in enumerate(filenames):
                            if os.path.basename(str(fp)) == bas:
                                found = k
                                break
                        if found >= 0:
                            file_idx = found

                # IMPORTANT: test against file_idx, not imgNum.
                label_file_indices = set(np.asarray(TimageLabels[10], dtype=int))
                if self.train and file_idx not in label_file_indices:
                    continue


                img = rio.open(imagePath)
                imageMatrix = img.read()
                
                max = get_img_sigma(imageMatrix[:, ::10, ::10])
                winSize = imageData['MS_winsize_pix']

            
                for i in range(0,len(TimageLabels[10])):
                    if TimageLabels[10][i]==file_idx:
                        row = imageLabels[i]
                        x,y = row[2:4].astype('int')
                        splitImg_np = imageMatrix[:, x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = to_ms_rowlist(row, splitImg_np)
                        if (splitImg_np.shape[1] == 0) or (splitImg_np.shape[2] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[8], "image source: ", rowlist[10])
                        else:
                            dataArray.append(rowlist)
                        #CST20240315print("data array", dataArray)
            elif len(TimageLabels) == 1: #testing
                    # If training, and there are no labeled split images from tiff image, skip loading it

                #CST 20240329
                # Determine file index for testing branch as well
                file_idx = imgNum
                if filenames is not None:
                    try:
                        file_idx = filenames.index(imagePath)
                    except ValueError:
                        bas = os.path.basename(imagePath)
                        found = -1
                        for k, fp in enumerate(filenames):
                            if os.path.basename(str(fp)) == bas:
                                found = k
                                break
                        if found >= 0:
                            file_idx = found

                # IMPORTANT: test against file_idx, not imgNum.
                if self.train and not any(row[10] == file_idx for row in TimageLabels[0]):
                    continue
                    

                img = rio.open(imagePath)
                imageMatrix = img.read()
                
                max = get_img_sigma(imageMatrix[:, ::10, ::10])
                winSize = imageData['MS_winsize_pix']
                #CST 20240329

                for i in range(0,len(TimageLabels[0])):
                    if TimageLabels[0][i][10] == file_idx:
                        row = imageLabels[i][0]
                        #print(row)
                        x,y = row[2:4].astype('int')
                        splitImg_np = imageMatrix[:,x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = to_ms_rowlist(row, splitImg_np)
                        if (splitImg_np.shape[1] == 0) or (splitImg_np.shape[2] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[8], "image source: ", rowlist[10])
                        else:
                            dataArray.append(rowlist)
                        
            else:
                print("Error with training or testing data")
            
        self.dataFrame = pd.DataFrame(dataArray, columns=['x_pix','y_pix','x_utm','y_utm','label','conf','img_source','img_mat'])




    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        splitImg_np = self.dataFrame.iloc[idx, 7]

        if self.transform:
            splitImg_np = self.transform(splitImg_np)
        splitImg_tensor = torch.from_numpy(splitImg_np)

        if self.train:
            label = int(self.dataFrame.iloc[idx, 4])
            return (splitImg_tensor, label)
        else:
            return splitImg_tensor

class SplitImageDatasetPAN(Dataset):

    def __init__(self, imgPath, imgData, labels, transform=None, train=False):
        """
        Multispectral split-image dataset.
        Uses ms_x/ms_y from the label rows so MS and pan cover the same UTM region.
        Expected labels (per row):
          [pan_x, pan_y, ms_x, ms_y, utm_x, utm_y, pan_label, pan_conf, ms_label, ms_conf, img_num]
        """
        self.train = train
        imagePaths = getImgPathsPan(imgPath)
        imageLabels = labels
        imageData = imgData
        self.transform = transform
        # If the saved dataset info includes the original filename ordering, use it
        filenames = None
        try:
            filenames = imageData.get('filename', None)
        except Exception:
            filenames = None

        dataArray = []
        
        def to_pan_rowlist(row, split_img):
            r = np.array(row)
            # Keep only PAN-relevant fields: pan_x, pan_y, utm_x, utm_y, pan_label, conf, img_source
            rowlist = [r[0], r[1], r[4], r[5], r[6], r[7], r[10]]
            rowlist.append(split_img)
            return rowlist
        for imgNum, imagePath in enumerate(imagePaths):
            # If training, and there are no labeled split images from tiff image, skip loading it
            TimageLabels = list(zip(*imageLabels)) #CST20240322 this may fail or not work as expected now
            if len(TimageLabels) == 11: #Training
                file_idx = imgNum
                if filenames is not None:
                    try:
                        file_idx = filenames.index(imagePath)
                    except ValueError:
                        bas = os.path.basename(imagePath)
                        found = -1
                        for k, fp in enumerate(filenames):
                            if os.path.basename(str(fp)) == bas:
                                found = k
                                break
                        if found >= 0:
                            file_idx = found

                # Convert the PAN file index to the paired MS file index used by the labels.
                label_file_indices = set(np.asarray(TimageLabels[10], dtype=int))

                if file_idx not in label_file_indices and filenames is not None:
                    pan_basename = os.path.basename(imagePath)
                    ms_basename = pan_basename.replace('-P1BS-', '-M1BS-')

                    paired_idx = next(
                        (
                            k for k, fp in enumerate(filenames)
                            if os.path.basename(str(fp)) == ms_basename
                        ),
                        None,
                    )

                    if paired_idx in label_file_indices:
                        file_idx = paired_idx
                    elif file_idx - 1 in label_file_indices:
                        file_idx -= 1

                if self.train and file_idx not in label_file_indices:
                    continue

                img = rio.open(imagePath)
                imageMatrix = img.read(1)
                
                max = get_img_sigma(imageMatrix[::10, ::10])
                winSize = imageData['winsize_pix']

                for i in range(0,len(TimageLabels[10])):
                    if TimageLabels[10][i]==file_idx:
                        row = imageLabels[i]
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = to_pan_rowlist(row, splitImg_np)
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[6], "image source: ", rowlist[10])
                        else:
                            dataArray.append(rowlist)
                        #CST20240315print("data array", dataArray)
            elif len(TimageLabels) == 1: #testing
                    # If training, and there are no labeled split images from tiff image, skip loading it

                #CST 20240329
                # Determine file index for testing branch as well
                file_idx = imgNum
                if filenames is not None:
                    try:
                        file_idx = filenames.index(imagePath)
                    except ValueError:
                        bas = os.path.basename(imagePath)
                        found = -1
                        for k, fp in enumerate(filenames):
                            if os.path.basename(str(fp)) == bas:
                                found = k
                                break
                        if found >= 0:
                            file_idx = found

                # Convert the PAN file index to the paired MS file index used by the labels.
                label_file_indices = {int(row[10]) for row in TimageLabels[0]}
                if file_idx not in label_file_indices and filenames is not None:
                    pan_basename = os.path.basename(imagePath)
                    ms_basename = pan_basename.replace('-P1BS-', '-M1BS-')

                    paired_idx = next(
                        (
                            k for k, fp in enumerate(filenames)
                            if os.path.basename(str(fp)) == ms_basename
                        ),
                        None,
                    )

                    if paired_idx in label_file_indices:
                        file_idx = paired_idx
                    elif file_idx - 1 in label_file_indices:
                        file_idx -= 1

                # IMPORTANT: test against file_idx, not imgNum.
                if self.train and not any(row[10] == file_idx for row in TimageLabels[0]):
                    continue
                    

                img = rio.open(imagePath)
                imageMatrix = img.read(1)
                
                max = get_img_sigma(imageMatrix[::10, ::10])
                winSize = imageData['winsize_pix']
                #CST 20240329

                for i in range(0,len(TimageLabels[0])):
                    if TimageLabels[0][i][10] == file_idx:
                        row = imageLabels[i][0]
                        #print(row)
                        x,y = row[0:2].astype('int')
                        splitImg_np = imageMatrix[x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = to_pan_rowlist(row, splitImg_np)
                        if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[6], "image source: ", rowlist[10])
                        else:
                            dataArray.append(rowlist)
                        
            else:
                print("Error with training or testing data")
            
        self.dataFrame = pd.DataFrame(dataArray, columns=['x_pix','y_pix','x_utm','y_utm','label','conf','img_source','img_mat'])




    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        splitImg_np = self.dataFrame.iloc[idx, 7]

        if self.transform:
            splitImg_np = self.transform(splitImg_np)
        splitImg_tensor = torch.from_numpy(splitImg_np)

        if self.train:
            label = int(self.dataFrame.iloc[idx, 4])
            return (splitImg_tensor, label)
        else:
            return splitImg_tensor

class MSPatchStatsDataset(Dataset):
    """
    Multispectral patch-statistics dataset for MLP classifiers.
    Computes per-band mean/std plus WRI mean/std per patch.
    """

    def __init__(self, imgPath, imgData, labels, wri_bands, stats_bands=None,
                 train=False, eps=1e-6):
        self.train = train
        imagePaths = getImgPathsMS(imgPath)
        imageLabels = labels
        imageData = imgData
        self.transform = transform
        # If the saved dataset info includes the original filename ordering, use it
        filenames = None
        try:
            filenames = imageData.get('filename', None)
        except Exception:
            filenames = None

        dataArray = []
        indices_list = []
        def to_ms_rowlist(row, split_img):
            r = np.array(row)
            # Keep only MS-relevant fields: ms_x, ms_y, utm_x, utm_y, ms_label, conf, img_source
            rowlist = [r[2], r[3], r[4], r[5], r[8], r[9], r[10]]
            rowlist.append(split_img)
            return rowlist
        for imgNum, imagePath in enumerate(imagePaths):
            # If training, and there are no labeled split images from tiff image, skip loading it
            TimageLabels = list(zip(*imageLabels)) #CST20240322 this may fail or not work as expected now
            a=0
            if len(TimageLabels) == 11: #Training
                for i in range(0,len(TimageLabels[10])):
                    if TimageLabels[10][i]==imgNum:
                        a=1
                if self.train and a == 0:
                            continue

                img = rio.open(imagePath)
                imageMatrix = img.read()
                
                max = get_img_sigma(imageMatrix[:, ::10, ::10])
                winSize = imageData['MS_winsize_pix']

                # Determine the file index used in the original dataset for this ms image (Fixes error of not labeling final GEOTIFF)
                file_idx = imgNum
                if filenames is not None:
                    try:
                        file_idx = filenames.index(imagePath)
                    except ValueError:
                        # Try matching by basename if full path formats differ
                        bas = os.path.basename(imagePath)
                        found = -1
                        for k,fp in enumerate(filenames):
                            if os.path.basename(fp) == bas:
                                found = k
                                break
                        if found >= 0:
                            file_idx = found

                for i in range(0,len(TimageLabels[10])):
                    if TimageLabels[10][i]==file_idx:
                        row = imageLabels[i]
                        x,y = row[2:4].astype('int')
                        splitImg_np = imageMatrix[:, x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = to_ms_rowlist(row, splitImg_np)
                        if (splitImg_np.shape[1] == 0) or (splitImg_np.shape[2] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[8], "image source: ", rowlist[10])
                        else:
                            dataArray.append(rowlist)
                        #CST20240315print("data array", dataArray)
            elif len(TimageLabels) == 1: #testing
                    # If training, and there are no labeled split images from tiff image, skip loading it

                #CST 20240329
                # Determine file index for testing branch as well
                file_idx = imgNum
                if filenames is not None:
                    try:
                        file_idx = filenames.index(imagePath)
                    except ValueError:
                        bas = os.path.basename(imagePath)
                        found = -1
                        for k,fp in enumerate(filenames):
                            if os.path.basename(fp) == bas:
                                found = k
                                break
                        if found >= 0:
                            file_idx = found

                for i in range(0,len(TimageLabels[0])):
                    if TimageLabels[0][i][10]==file_idx:
                        a=1
                if self.train and a == 0:
                            continue
                    

                img = rio.open(imagePath)
                imageMatrix = img.read()
                
                max = get_img_sigma(imageMatrix[:, ::10, ::10])
                winSize = imageData['MS_winsize_pix']
                #CST 20240329
                # Determine file index for this ms image for testing
                file_idx = imgNum
                if filenames is not None:
                    try:
                        file_idx = filenames.index(imagePath)
                    except ValueError:
                        bas = os.path.basename(imagePath)
                        found = -1
                        for k,fp in enumerate(filenames):
                            if os.path.basename(fp) == bas:
                                found = k
                                break
                        if found >= 0:
                            file_idx = found

                for i in range(0,len(TimageLabels[0])):
                    if TimageLabels[0][i][10] == file_idx:
                        row = imageLabels[i][0]
                        #print(row)
                        x,y = row[2:4].astype('int')
                        splitImg_np = imageMatrix[:,x:x+winSize[0],y:y+winSize[1]]
                        splitImg_np = scaleImage(splitImg_np, max)
                        rowlist = to_ms_rowlist(row, splitImg_np)
                        if (splitImg_np.shape[1] == 0) or (splitImg_np.shape[2] == 0):
                            print("Error with an image: ", i, "class: ", rowlist[4], "image source: ", rowlist[6])
                        else:
                            dataArray.append(rowlist)
                        
            else:
                print("Error with training or testing data")
            
        self.dataFrame = pd.DataFrame(dataArray, columns=['x_pix','y_pix','x_utm','y_utm','label','conf','img_source','img_mat'])

    def _compute_features(self, patch, g_idx, r_idx, nir_idx, mir_idx):
        feats = []
        for b in self.stats_bands:
            band = patch[b]
            feats.append(float(band.mean()))
            feats.append(float(band.std()))

        green = patch[g_idx]
        red = patch[r_idx]
        nir = patch[nir_idx]
        mir = patch[mir_idx]
        denom = nir + mir
        wri = (green + red) / (denom + self.eps)
        feats.append(float(wri.mean()))
        feats.append(float(wri.std()))

        return feats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.features[idx], dtype=torch.float32)
        if x.ndim == 0:
            x = x.view(1)
        y = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return x, y


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
