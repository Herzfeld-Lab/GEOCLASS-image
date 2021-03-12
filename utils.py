import os
import utm
import numpy as np
import cv2
import math
import xml.etree.ElementTree as ET
import rasterio as rio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageOps
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
from netCDF4 import Dataset
import glob
import argparse
import pyproj
from pyproj import Transformer
from pyproj import CRS

def to_netCDF(data, filepath):

    attributes = data[0]

    array = data[1]
    height,width = array.shape

    out = Dataset(filepath+'.nc','w')

    out.createDimension('img_x', height)
    out.createDimension('img_y', height)
    out.createDimension('UTM_x', height)
    out.createDimension('UTM_y', height)
    out.createDimension('class', height)
    out.createDimension('conf', height)

    img_x = out.createVariable('img_x','f8',('img_x'))
    img_y = out.createVariable('img_y','f8',('img_y'))
    UTM_x = out.createVariable('UTM_x','f8',('UTM_x'))
    UTM_y = out.createVariable('UTM_y','f8',('UTM_y'))
    type = out.createVariable('class','i1',('img_x'))
    conf = out.createVariable('conf','f8',('img_x'))

    img_x[:] = array[:,0]
    img_y[:] = array[:,1]
    UTM_x[:] = array[:,2]
    UTM_y[:] = array[:,3]
    type[:] = array[:,4]
    conf[:] = array[:,5]

    out.setncatts(attributes)
    out.close()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_img_sigma(img_mat):
    std = np.std(img_mat)
    max = img_mat.max()
    mean = np.mean(img_mat)
    if max > mean+2.5*std:
        return (mean + 2.5*std)
    else:
        return (img_mat.max())

def scaleImage(img, max):
    img = img/max
    img = img * 255
    img[img > 255] = 255
    return np.ceil(img).astype('uint8')

def getImgPaths(topDir):
    return glob.glob(topDir + '/*.tif')

def utm_to_pix(imgSize,utmBounds,utmCoord):
    """
    Scales UTM coordinates to fit within a given satellite background image's
    pixel space

    Args:
        imgSize (nparray):      Size of image in pixels
        utmBounds (nparray):    Boundaries of image in UTM (coordinates of top left and bottom right)
        utmCoord: (nparray):    List of UTM coordinates to scale within image boundaries
    Returns:
        Scaled list of coordinates in image space (pixel space)
    """
    # Create closed polygon by appending first element to end of list
    utmCoord = np.append(utmCoord, [utmCoord[0]], axis=0)

    # Find range of image's UTM bounds
    utmRange = np.max(utmBounds,1) - np.min(utmBounds,1)

    # Find step in UTM units corresponding to one pixel in image space
    step = imgSize.astype(float) / utmRange.astype(float)

    # Scale UTM coordinates by step to find equivalent pixel-space location
    pixCoord = (utmCoord - np.min(utmBounds,1)) * step
    return pixCoord.astype(int)

def utm_to_pix2(imgSize,utmBounds,utmCoord):
    """
    Scales UTM coordinates to fit within a given satellite background image's
    pixel space

    Args:
        imgSize (nparray):      Size of image in pixels
        utmBounds (nparray):    Boundaries of image in UTM (coordinates of top left and bottom right)
        utmCoord: (nparray):    List of UTM coordinates to scale within image boundaries
    Returns:
        Scaled list of coordinates in image space (pixel space)
    """
    # Create closed polygon by appending first element to end of list
    utmCoord = np.append(utmCoord, [utmCoord[0]], axis=0)

    # Find range of image's UTM bounds
    utmRange = np.max(utmBounds,1) - np.min(utmBounds,1)
    print(utmRange)

    # Find step in UTM units corresponding to one pixel in image space
    step = imgSize.astype(float) / utmRange.astype(float)
    print(step)

    # Scale UTM coordinates by step to find equivalent pixel-space location
    pixCoord = (utmCoord - np.min(utmBounds,1)) * step
    return pixCoord.astype(int)

def xml_to_latlon(xmlPath):
    xmlTree = ET.parse(xmlPath)

    if xmlTree.findall('.//BAND_P'):
        UL = (float(xmlTree.findall('.//BAND_P/ULLAT')[0].text),float(xmlTree.findall('.//BAND_P/ULLON')[0].text))
        UR = (float(xmlTree.findall('.//BAND_P/URLAT')[0].text),float(xmlTree.findall('.//BAND_P/URLON')[0].text))
        LL = (float(xmlTree.findall('.//BAND_P/LLLAT')[0].text),float(xmlTree.findall('.//BAND_P/LLLON')[0].text))
        LR = (float(xmlTree.findall('.//BAND_P/LRLAT')[0].text),float(xmlTree.findall('.//BAND_P/LRLON')[0].text))
        GSD = (float(xmlTree.findall('.//IMAGE/MEANCOLLECTEDROWGSD')[0].text),float(xmlTree.findall('.//IMAGE/MEANCOLLECTEDCOLGSD')[0].text))
    else:
        UL = (float(xmlTree.findall('.//ULLAT')[0].text),float(xmlTree.findall('.//ULLON')[0].text))
        UR = (float(xmlTree.findall('.//URLAT')[0].text),float(xmlTree.findall('.//URLON')[0].text))
        LL = (float(xmlTree.findall('.//LLLAT')[0].text),float(xmlTree.findall('.//LLLON')[0].text))
        LR = (float(xmlTree.findall('.//LRLAT')[0].text),float(xmlTree.findall('.//LRLON')[0].text))
        #GSD = (float(xmlTree.findall('./IMAGE/MEANCOLLECTEDROWGSD')[0].text),float(xmlTree.findall('./IMAGE/MEANCOLLECTEDCOLGSD')[0].text))

    latlon = [UL, UR, LR, LL]

    return latlon

def get_geotiff_bounds(geotiff, epsg_code):
    # Get transform from tiff image georeferencing data to UTM
    crs_in = CRS.from_wkt(geotiff.crs.wkt)
    crs_out = CRS.from_epsg(epsg_code)
    transform = Transformer.from_crs(crs_in, crs_out)

    ul_in = geotiff.transform*(0, 0)
    lr_in = geotiff.transform*(geotiff.width, geotiff.height)
    ll_in = geotiff.transform*(0, geotiff.height)
    ur_in = geotiff.transform*(geotiff.width, 0)

    ul_out = np.array(transform.transform(ul_in[0],ul_in[1]))
    lr_out = np.array(transform.transform(lr_in[0],lr_in[1]))
    ll_out = np.array(transform.transform(ll_in[0],ll_in[1]))
    ur_out = np.array(transform.transform(ur_in[0],ur_in[1]))

    # Get UTM boundaries
    bbox = np.array((ul_out,ur_out,lr_out,ll_out))
    return bbox

def plot_geotif_bbox(xmlPath, contourPath, bgImgPath, bgUTMPath):
    """
    Plots an overlay of a geotiff image onto an image of the relevant glacier.

    Args:
        xmlPath (str):      Path to xml file corresponding to tif image
        contourPath (str):  Path to UTM coordinate contour of glacier border in npy format.
        bgImgPath (str):    Path to background image of glacier
        bgUTMPath (str):    Path to npy file with UTM coordinates of background image
    Returns:
        None
    """

    # Load background image and corresponding UTM boundaries
    bgImg = cv2.imread(bgImgPath)
    bgImgUTM = np.load(bgUTMPath)

    # Find image size in pixels
    height,width,channels = bgImg.shape
    imgSize = np.array([width,height])

    # Load UTM coordinates of glacier contour and translate to pixel locations
    contourUTM = np.load(contourPath)
    contourPixel = utm_to_pix(imgSize, bgImgUTM, contourUTM)

    # Load coordinaates of TIF image from xml file, translate to pixel locations
    bboxLatlon = xml_to_latlon(xmlPath)
    bboxUTM = np.array([utm.from_latlon(i[0],i[1])[0:2] for i in bboxLatlon])
    bboxPixel = utm_to_pix(imgSize, bgImgUTM, bboxUTM)

    # Draw glacier contour onto background image
    for i in range(len(contourPixel)-1):
        cv2.line(bgImg, tuple(contourPixel[i]), tuple(contourPixel[i+1]), (0,0,255), 3)

    # Draw TIF image  boundaries onto background image
    for i in range(len(bboxUTM)):
        cv2.line(bgImg, tuple(bboxPixel[i]), tuple(bboxPixel[i+1]), (255,0,0), 3)

    # Flip image to display in correct orientation
    bgImg = cv2.flip(bgImg,0)

    # Display image and wait for user input
    #cv2.namedWindow(bgImgPath[:-4], cv2.WINDOW_NORMAL)
    cv2.imwrite(bgImgPath[:-4]+'.jpg', bgImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def directional_vario(img, numLag, lagThresh = 0.8):
    """
    Implements the directional vario function in python. The variogram is computed
    in 4 different directions: North/South, East/West, and the two diagonals. The
    results are concatenated together as the rows of a numpy array and returned

    Args:
        img (nparray):      Image matrix to compute vario function on
        numLag (int):       The number of different lag values to calculate variogram with
        lagThresh (float):  Threshold for maximum lag value as a percentage of smallest
                            image dimension
    Returns:
        nparray:            Array containing the directional variogram for each
                            lag value for all 4 directions.
    """

    # If numLag is greater than smalles image dimension * lagThresh, ovverride
    imSize = img.shape
    minDim = min(imSize)
    imRange = minDim * lagThresh
    lagStep = math.floor(imRange / numLag)

    vario = np.zeros((4,numLag))

    # For each value of lag, calculate directional variogram in given direction
    for i,h in enumerate(range(1,numLag*lagStep,lagStep)):

        # North/South direction
        diff = img[0:-h,:minDim] - img[h:,:minDim]
        numPairs = np.prod(diff.shape)
        v_h = (1. / numPairs) * np.sum(diff**2)
        vario[0,i] = v_h

        # East/West direction
        diff = img[:,0:minDim-h] - img[:,h:minDim]
        numPairs = np.prod(diff.shape)
        v_h = (1. / numPairs) * np.sum(diff**2)
        vario[1,i] = v_h

        # Approximate h in diagonal direction by dividing by tangent
        h_diag = int(round(h / math.sqrt(2)))

        # Diagonal (top left to bottom right)
        diff = img[0:minDim-h_diag,0:minDim-h_diag] - img[h_diag:minDim,h_diag:minDim]
        numPairs = np.prod(diff.shape)
        v_h = (1. / numPairs) * np.sum(diff**2)
        vario[2,i] = v_h

        # Diagonal (bottom left to top right)
        diff = img[h_diag:minDim,0:minDim-h_diag] - img[0:minDim-h_diag,h_diag:minDim]
        numPairs = np.prod(diff.shape)
        v_h = (1. / numPairs) * np.sum(diff**2)
        vario[3,i] = v_h

    return vario

def batch_directional_vario(img_arr, numLag, lagThresh = 0.8):

    ret = np.zeros((img_arr.shape[0],4,numLag))

    for i,img in enumerate(img_arr):

        ret[i] = directional_vario(img,numLag,lagThresh)

    return ret

def batch_rotate_vario(vario):
    ret = np.zeros((vario.shape[0],vario.shape[2]*3))

    for i in range(vario.shape[0]):
        rand = random.uniform(0,1)
        v = vario[i]
        if rand < 0.25:
            ret[i,:] = np.concatenate((v[0,:],v[1,:],v[2,:]))
        elif rand < 0.5:
            ret[i,:] = np.concatenate((v[1,:],v[0,:],v[2,:]))
        elif rand < 0.75:
            ret[i,:] = np.concatenate((v[0,:],v[1,:],v[3,:]))
        elif rand < 1:
            ret[i,:] = np.concatenate((v[1,:],v[0,:],v[3,:]))

    return ret

def generate_config(yaml_obj):
    config_str = '''
### MODEL PARAMETERS ###

model:          {}
num_classes:    {}
vario_num_lag:  {}
hidden_layers:  {}
activation:     {}

### DATASET PARAMETERS ###

img_path:           {}
npy_path:           {}
train_path:         {}
valid_path:         {}
class_enum:         {}
utm_epsg_code:      {}
split_img_size:     {}
train_test_split:   {}

### TRAINING PARAMETERS ###

use_cuda:       {}
num_epochs:     {}
learning_rate:  {}
batch_size:     {}
optimizer:      {}

### DATA AUGMENTATION PARAMETERS ###

directional_vario:  {}
random_rotate:      {}
random_shift:       {}
random_contrast:    {}
random_distort:     {}

### VISUALIZATION PARAMETERS ###

contour_path:       {}
custom_color_map:   {}
bg_img_path:        {}
bg_UTM_path:        {}
        '''.format(yaml_obj['model'],
                   yaml_obj['num_classes'],
                   yaml_obj['vario_num_lag'],
                   yaml_obj['hidden_layers'],
                   yaml_obj['activation'],
                   yaml_obj['img_path'],
                   yaml_obj['npy_path'],
                   yaml_obj['train_path'],
                   yaml_obj['valid_path'],
                   yaml_obj['class_enum'],
                   yaml_obj['utm_epsg_code'],
                   yaml_obj['split_img_size'],
                   yaml_obj['train_test_split'],
                   yaml_obj['use_cuda'],
                   yaml_obj['num_epochs'],
                   yaml_obj['learning_rate'],
                   yaml_obj['batch_size'],
                   yaml_obj['optimizer'],
                   yaml_obj['directional_vario'],
                   yaml_obj['random_rotate'],
                   yaml_obj['random_shift'],
                   yaml_obj['random_contrast'],
                   yaml_obj['random_distort'],
                   yaml_obj['contour_path'],
                   yaml_obj['custom_color_map'],
                   yaml_obj['bg_img_path'],
                   yaml_obj['bg_UTM_path'])

    return config_str
