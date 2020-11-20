import os
import utm
import numpy as np
from os import path
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
import cv2
from PIL import Image

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Load config parameters
imgPath = cfg['img_path']
winSize = cfg['split_img_size']
contourPath = cfg['contour_path']
epsgCode = cfg['utm_epsg_code']
classEnum = cfg['class_enum']

# Load tiff image
print("Loading Tiff Image...")
tiffImg = rio.open(imgPath)
band1 = tiffImg.read(1)
imgSize = band1.shape

if tiffImg.transform == Affine.identity():
    xmlPath = imgPath[:-4] + '.xml'
else:
    xmlPath = None


# Load UTM glacier contour
print("Loading Glacier Contour...")
contourUTM = np.load(contourPath)
contourPolygon = Polygon(contourUTM)

# print(contourUTM)

# Get geo transform for tiff image
if xmlPath:
    crs_latlon = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(epsgCode)
    transform = Transformer.from_crs(crs_latlon, crs_utm)

else:
    crs_in = CRS.from_wkt(tiffImg.crs.to_string())
    crs_utm = CRS.from_epsg(epsgCode)
    transform = Transformer.from_crs(crs_in, crs_utm)
    print(transform.to_wkt())

# Calculate image bounding box in UTM
if xmlPath:
    bboxLatlon = xml_to_latlon(xmlPath)
    bboxUTM = [0]*len(bboxLatlon)
    for i in range(len(bboxLatlon)):
        bboxUTM[i] = transform.transform(bboxLatlon[i][0], bboxLatlon[i][1])

    bboxUTM = np.array(bboxUTM)

else:
    ul_in = tiffImg.transform*(0, 0)
    lr_in = tiffImg.transform*(tiffImg.width, tiffImg.height)
    ll_in = tiffImg.transform*(0, tiffImg.height)
    ur_in = tiffImg.transform*(tiffImg.width, 0)

    ul_out = np.array(transform.transform(ul_in[0],ul_in[1]))
    lr_out = np.array(transform.transform(lr_in[0],lr_in[1]))
    ll_out = np.array(transform.transform(ll_in[0],ll_in[1]))
    ur_out = np.array(transform.transform(ur_in[0],ur_in[1]))

    bboxUTM = np.array((ul_out,ur_out,lr_out,ll_out))

#print('bboxUTM', bboxUTM)

image_UL_UTM = np.array([bboxUTM[:,0].min(),bboxUTM[:,1].max()])
image_LR_UTM = np.array([bboxUTM[:,0].max(),bboxUTM[:,1].min()])
image_UR_UTM = np.array([bboxUTM[:,0].min(),bboxUTM[:,1].min()])
image_LL_UTM = np.array([bboxUTM[:,0].max(),bboxUTM[:,1].max()])

LL_img = bboxUTM[0]
LR_img = bboxUTM[1]
UR_img = bboxUTM[2]
UL_img = bboxUTM[3]

UL_max = bboxUTM.min(axis = 0)
LR_max = bboxUTM.max(axis = 0)
UR_max = np.array([LR_max[0], UL_max[1]])
LL_max = np.array([UL_max[0], LR_max[1]])

UTM_north = bboxUTM[:,0]
UTM_east = bboxUTM[:,1]

UTM_north = np.sort(UTM_north)[1:3]
UTM_east = np.sort(UTM_east)[1:3]

UL_min = UL_img
LR_min = np.array([UTM_north.max(axis = 0), LR_max[1]])
UR_min = np.array([LR_min[0], UL_min[1]])
LL_min = np.array([UL_min[0], LR_min[1]])

print('UL_img',UL_img)
print('UR_img',UR_img)
print('LR_img',LR_img)
print('LL_img',LL_img)

print('\n')

print('UL_max',UL_max)
print('LR_max',LR_max)
print('UR_max',UR_max)
print('LL_max',LL_max)

print('\n')

print('UL_min',UL_min)
print('LR_min',LR_min)
print('UR_min',UR_min)
print('LL_min',LL_min)

print('\n')

UL = np.array([UTM_north.min(axis=0), UTM_east.min(axis=0)])
LR = np.array([UTM_north.max(axis=0), UTM_east.max(axis=0)])

LL = np.array([LR[0], UL[1]])
UR = np.array([UL[0], LR[1]])

UTM_bounds = np.array([LR,UL])

utm_range = LR_min - UL_min
pixSizes = utm_range / np.flip(np.array(imgSize))

UTM_width, UTM_height = np.abs(UL - LR)[0], np.abs(UL - LR)[1]

rot_angle_x = angle_between(UR_img - UL_img, [-1,0])*180/math.pi
rot_angle_y = angle_between(UR_img - LR_img, [0,-1])*180/math.pi

print('ANGLE: ', rot_angle_y)

y_shear = (LR_img[1] - LR_max[1]) / (UR_max[0] - UL_max[0])

x_shear = (LL_img[0] - LL_min[0]) / (LL_max[1] - UL_max[1])

utm_affine_transform = Affine(pixSizes[0], 0, UL_min[0],
                       0, pixSizes[1], UL_min[1])

utm_affine_transform = utm_affine_transform * Affine.shear(rot_angle_y, rot_angle_x)


ULmaybe = utm_affine_transform * (0,0)
LRmaybe = utm_affine_transform * (imgSize[0], imgSize[1])
print('UL MAYBE: ', ULmaybe)
print('LR MAYBE: ', LRmaybe)

#y_shear = angle_between(image_UR_UTM - image_UL_UTM, [1,0])*180/math.pi
#x_shear = angle_between(image_LL_UTM - image_UL_UTM, [0,1])*180/math.pi
print('Y SHEAR: ', y_shear)
print('X SHEAR: ', x_shear)


#utm_affine_transform = rio.transform.from_bounds(bboxUTM.max(axis = 0)[0], bboxUTM.min(axis = 0)[1], bboxUTM.min(axis = 0)[0], bboxUTM.max(axis = 0)[1], tiffImg.width, tiffImg.height)
#utm_affine_transform = Affine(utm_affine_transform.a, 0.1, utm_affine_transform.c,
#                       0.1, utm_affine_transform.e, utm_affine_transform.f)

print('IMAGE TRANSFORM')
print(utm_affine_transform)

#print(tiffImg.transform)
    #print(img_transform*(0,0),'\t---\t',image_UL_UTM)
    #print(img_transform*imgSize,'\t---\t',image_LR_UTM)

pix_coords_list = []

print('Splitting Image...')

for row,i in enumerate(range(0,imgSize[0],winSize[0])):

    for col,j in enumerate(range(0,imgSize[1],winSize[1])):

        UL_pix = (i,j)
        LR_pix = (i+winSize[0],j+winSize[1])

        if xmlPath:
            UL_UTM = np.asarray(utm_affine_transform*(UL_pix[1],imgSize[0] - UL_pix[0]))
            LR_UTM = np.asarray(utm_affine_transform*(LR_pix[1],imgSize[0] - LR_pix[0]))

            #UL_UTM = np.flip(UL_UTM)
            #LR_UTM = np.flip(LR_UTM)

            #print(UL_UTM)

        else:
            '''
            UL_stere = transform*(UL_pix[1],UL_pix[0])
            LR_stere = transform*(LR_pix[1],LR_pix[0])

            UL_latlon = stere_to_latlon.transform(UL_stere[0],UL_stere[1])
            LR_latlon = stere_to_latlon.transform(LR_stere[0],LR_stere[1])

            UL_UTM = latlon_to_utm.transform(UL_latlon[0],UL_latlon[1])
            LR_UTM = latlon_to_utm.transform(LR_latlon[0],LR_latlon[1])
            '''
            ul_in = tiffImg.transform*(UL_pix[1],UL_pix[0])
            lr_in = tiffImg.transform*(LR_pix[1],LR_pix[0])

            UL_UTM = np.array(transform.transform(ul_in[0],ul_in[1]))
            LR_UTM = np.array(transform.transform(lr_in[0],lr_in[1]))

            #UL_UTM_test = np.asarray(utm_affine_transform*(UL_pix[1],UL_pix[0]))
            #LR_UTM_test = np.asarray(utm_affine_transform*(LR_pix[1],LR_pix[0]))

        splitImg = band1[i:i+winSize[0],j:j+winSize[1]]

        if splitImg.shape != tuple(winSize):
            continue

        if Point(UL_UTM[0],UL_UTM[1]).within(contourPolygon) and Point(LR_UTM[0],LR_UTM[1]).within(contourPolygon):
            if splitImg[splitImg==0].shape[0] == 0:
                pix_coords_list.append([i,j,UL_UTM[0],UL_UTM[1],-1,0])

print('Saving Dataset...')
pix_coords_np = np.array(pix_coords_list)

#print(pix_coords_np)

train_size = int(cfg['train_test_split'] * pix_coords_np.shape[0])

train_indeces = np.random.choice(range(pix_coords_np.shape[0]), train_size, replace=False)
test_indeces = list(set(range(pix_coords_np.shape[0])) - set(train_indeces))

#print(train_indeces)
#print(test_indeces)

# extract your samples:
train_coords = pix_coords_np[train_indeces, :]
test_coords = pix_coords_np[test_indeces, :]

if xmlPath:
    transform = utm_affine_transform
else:
    transform = None

info = {'filename': imgPath,
        'contour_path': contourPath,
        'winsize_pix': winSize,
        #'winsize_utm': UTM_winSize,
        'transform': transform,
        'class_enumeration': classEnum}

print('FINAL INFO: ', info)

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
print('Created %d split images'%(pix_coords_np.shape[0]))

print('Saved Full Dataset to ', dataset_path)



f = open(args.config, 'w')
f.write(generate_config(cfg))
f.close()
