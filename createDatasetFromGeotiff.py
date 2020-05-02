import os
import utm
import numpy as np
#import cv2
import math
import xml.etree.ElementTree as ET
from utils import *
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import rasterio as rio
from rasterio.transform import Affine
import pandas as pd
import pyproj
from pyproj import Transformer
from pyproj import CRS
import yaml
import argparse

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

imgPath = cfg['img_path']
winSize = cfg['split_img_size']
contourPath = cfg['contour_path']
epsgCode = cfg['utm_epsg_code']
classEnum = cfg['class_enum']
xmlPath = None

print(winSize[0])

# Load tiff image
print("Loading Tiff Image...")
tiffImg = rio.open(imgPath)
band1 = tiffImg.read(1)
imgSize = band1.shape

# Load UTM glacier contour
print("Loading Glacier Contour...")
contourUTM = np.load(contourPath)
contourPolygon = Polygon(contourUTM)

# Create Affine transform for tiff image
transform = tiffImg.transform
UL_stere = transform*(0,0)

'''
if xmlPath:
    crs_stere = CRS.from_epsg(4326)
    crs_latlon = CRS.from_epsg(4326)
    #crs_stere = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(epsgCode)

else:
'''
crs_stere = CRS.from_wkt(tiffImg.crs.to_string())
crs_latlon = CRS.from_epsg(4326)
crs_utm = CRS.from_epsg(epsgCode)

print(crs_stere)

stere_to_latlon = Transformer.from_crs(crs_stere, crs_latlon)#, always_xy=True)
latlon_to_utm = Transformer.from_crs(crs_latlon, crs_utm)
#UL_utm = t.transform(UL_stere[1],UL_stere[0])
#transform = Affine(transform[0],transform[1],UL_utm[0],transform[3],transform[4],UL_utm[1])

if xmlPath:
    bboxLatlon = xml_to_latlon(xmlPath)
    bboxUTM = [0]*len(bboxLatlon)
    for i in range(len(bboxLatlon)):
        bboxUTM[i] = latlon_to_utm.transform(bboxLatlon[i][0], bboxLatlon[i][1])

    bboxUTM = np.array(bboxUTM)

    print('bboxUTM', bboxUTM)

    image_UL_UTM = np.array([bboxUTM[:,0].max(),bboxUTM[:,1].min()])
    image_LR_UTM = np.array([bboxUTM[:,0].min(),bboxUTM[:,1].max()])

    pix_size = image_LR_UTM - image_UL_UTM

    pix_size = abs(pix_size / np.array(imgSize))

    print(image_UL_UTM, bboxUTM[0])

    img_transform = Affine.from_gdal(image_UL_UTM[1], pix_size[1], 0.0, image_UL_UTM[0], 0.0, -pix_size[0])

    print(img_transform)

    print(img_transform*(0,0),'\t---\t',image_UL_UTM)
    print(img_transform*imgSize,'\t---\t',image_LR_UTM)

pix_coords_list = []

print('Splitting Image...')

for row,i in enumerate(range(0,imgSize[0],winSize[0])):

    for col,j in enumerate(range(0,imgSize[1],winSize[1])):

        UL_pix = (i,j)
        LR_pix = (i+winSize[0],j+winSize[1])
        '''
        if xmlPath:
            UL_UTM = np.asarray(img_transform*(UL_pix[1],UL_pix[0]))
            LR_UTM = np.asarray(img_transform*(LR_pix[1],LR_pix[0]))

            UL_UTM = np.flip(UL_UTM)
            LR_UTM = np.flip(LR_UTM)

        else:
        '''
        UL_stere = transform*(UL_pix[1],UL_pix[0])
        LR_stere = transform*(LR_pix[1],LR_pix[0])

        UL_latlon = stere_to_latlon.transform(UL_stere[0],UL_stere[1])
        LR_latlon = stere_to_latlon.transform(LR_stere[0],LR_stere[1])

        UL_UTM = latlon_to_utm.transform(UL_latlon[0],UL_latlon[1])
        LR_UTM = latlon_to_utm.transform(LR_latlon[0],LR_latlon[1])

        splitImg = band1[i:i+winSize[0],j:j+winSize[1]]

        if splitImg.shape != tuple(winSize):
            continue

        if Point(UL_UTM[0],UL_UTM[1]).within(contourPolygon) and Point(LR_UTM[0],LR_UTM[1]).within(contourPolygon):
            if splitImg[splitImg==0].shape[0] == 0:
                pix_coords_list.append([i,j,UL_UTM[0],UL_UTM[1],-1,0])

print('Saving Dataset...')
pix_coords_np = np.array(pix_coords_list)

train_size = int(cfg['train_test_split'] * pix_coords_np.shape[0])

train_indeces = np.random.choice(range(pix_coords_np.shape[0]), train_size, replace=False)
test_indeces = list(set(range(pix_coords_np.shape[0])) - set(train_indeces))

# extract your samples:
train_coords = pix_coords_np[train_indeces, :]
test_coords = pix_coords_np[test_indeces, :]

print(pix_coords_np.shape)
print(train_coords.shape)
print(test_coords.shape)

info = {'filename': imgPath,
        'contour_path': contourPath,
        'winsize_pix': winSize,
        #'winsize_utm': UTM_winSize,
        'transform': transform,
        'class_enumeration': classEnum}

save_array_full = np.array([info, pix_coords_np])
#save_array_train = np.array([info, train_coords])
#save_array_test = np.array([info, test_coords])

dataset_path = imgPath[:-4] + "_(%d,%d)_split"%(winSize[0],winSize[1])
cfg['txt_path'] = dataset_path+'.npy'
#train_path = '/'.join(args.config.split('/')[:-1]) + 'train'
#test_path = '/'.join(args.config.split('/')[:-1]) + 'test'

np.save(dataset_path, save_array_full)
#np.save(train_path, save_array_train)
#np.save(test_path, save_array_test)

print('Done!')
print('Saved Full Dataset to ',imgPath[:-4] + "_(%d,%d)_dataset"%(winSize[0],winSize[1]))

f = open(args.config, 'w')
f.write(generate_config(cfg))
f.close()
