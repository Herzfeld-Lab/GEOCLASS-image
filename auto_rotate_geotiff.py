import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore
from PIL.ImageQt import ImageQt
from utils import *
from Models import *
from Dataset import *
import rasterio as rio
import numpy as np
import utm
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import pyproj
from pyproj import Transformer
from pyproj import CRS
from affine import Affine

from scipy.ndimage import rotate

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

def rot(image, xy, angle):
    im_rot = rotate(image,angle)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    a = np.deg2rad(angle)

    org = np.zeros((4,2))
    new = np.zeros((4,2))

    for i in range(len(xy)):
        org[i] = np.array(xy[i])-org_center
        news = np.array([org[i][0]*np.cos(a) + org[i][1]*np.sin(a),-org[i][0]*np.sin(a) + org[i][1]*np.cos(a)])
        new[i] = news+rot_center
    return new

def cv2_shear(img_mat, transform):
    #Parameters of the affine transform:
    angle = 45; #Angle in degrees.
    shear = 1;
    translation = 5;

    type_border = cv2.BORDER_CONSTANT;
    color_border = (255,255,255);

    original_image = cv2.imread(name_image_file);
    rows,cols,ch = original_image.shape;

    #First: Necessary space for the rotation
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1);
    cos_part = np.abs(M[0, 0]); sin_part = np.abs(M[0, 1]);
    new_cols = int((rows * sin_part) + (cols * cos_part));
    new_rows = int((rows * cos_part) + (cols * sin_part));

    #Second: Necessary space for the shear
    new_cols += (shear*new_cols);
    new_rows += (shear*new_rows);

    #Calculate the space to add with border
    up_down = int((new_rows-rows)/2); left_right = int((new_cols-cols)/2);

    final_image = cv2.copyMakeBorder(original_image, up_down, up_down,left_right,left_right,type_border, value = color_border);
    rows,cols,ch = final_image.shape;

    #Application of the affine transform.
    M_rot = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1);
    translat_center_x = -(shear*cols)/2;
    translat_center_y = -(shear*rows)/2;

def auto_rotate_geotiff(tiffInfo, tiffImg, img_mat, epsg_code, contourUTM, tiff_num):

    if tiffInfo['transform'][tiff_num]:
        transform = tiffInfo['transform'][tiff_num]

        '''
        ul_out = np.array(transform*(0, 0))
        lr_out = np.array(transform*(tiffImg.width, tiffImg.height))
        ll_out = np.array(transform*(0, tiffImg.height))
        ur_out = np.array(transform*(tiffImg.width, 0))
        '''
        ul_out = np.array(transform*(0, 0))
        lr_out = np.array(transform*(tiffImg.height, tiffImg.width))
        ll_out = np.array(transform*(0, tiffImg.width))
        ur_out = np.array(transform*(tiffImg.height, 0))
        '''
        bbox = np.array((ul_out,ur_out,lr_out,ll_out))

        UL = bbox.min(axis = 0)
        LR = bbox.max(axis = 0)

        height,width,channels = img_mat.shape
        imgSize = np.array([height,width])


        img_mat_rot = img_mat
        UTM_bounds = np.array([LR,UL])

        utm_range = LR - UL
        pixSizes = utm_range / imgSize

        transform_rot = Affine(pixSizes[0], 0, UL[0],
                               0, pixSizes[1], UL[1])

        contourPixel = utm_to_pix(imgSize, UTM_bounds.T, contourUTM)

        img_mat_rot = cv2.flip(img_mat_rot,0)

        for i in range(len(contourPixel)-1):
            cv2.line(img_mat_rot, tuple(contourPixel[i]), tuple(contourPixel[i+1]), (0,0,255), 2)

        #for i in range(len(rots)):
        #    cv2.circle(img_mat_rot,(int(rots[i][0]),int(rots[i][1])),8,(255,0,0),thickness=-1)

        img_mat_rot = cv2.flip(img_mat_rot,0)
        '''
        bbox = np.array((ul_out,ur_out,lr_out,ll_out))

        UL = bbox.min(axis = 0)
        LR = bbox.max(axis = 0)

        rot_angle = angle_between(ur_out - ul_out, [1,0])*180/math.pi

        orig = [[0,0],
                [img_mat.shape[1],0],
                [0,img_mat.shape[0]],
                [img_mat.shape[1],img_mat.shape[0]]]

        img_mat_rot = rotate(img_mat,-rot_angle)

        rots = rot(img_mat,orig,-rot_angle)

        height,width,channels = img_mat_rot.shape
        imgSize = np.array([height,width])

        UTM_bounds = np.array([LR,UL])

        utm_range = LR - UL
        pixSizes = utm_range / imgSize

        transform_rot = Affine(pixSizes[0], 0, UL[0],
                               0, pixSizes[1], UL[1])

        print(transform)
        print(transform_rot)

        contourPixel = utm_to_pix(imgSize, UTM_bounds.T, contourUTM)

        img_mat_rot = cv2.flip(img_mat_rot,0)

        for i in range(len(contourPixel)-1):
            cv2.line(img_mat_rot, tuple(contourPixel[i]), tuple(contourPixel[i+1]), (0,0,255), 2)

        #for i in range(len(rots)):
        #    cv2.circle(img_mat_rot,(int(rots[i][0]),int(rots[i][1])),8,(255,0,0),thickness=-1)

        img_mat_rot = cv2.flip(img_mat_rot,0)

        print(img_mat_rot.shape)

    else:
        crs_in = CRS.from_wkt(tiffImg.crs.wkt)
        crs_out = CRS.from_epsg(epsg_code)
        transform = Transformer.from_crs(crs_in, crs_out)

        ul_in = tiffImg.transform*(0, 0)
        lr_in = tiffImg.transform*(tiffImg.width, tiffImg.height)
        ll_in = tiffImg.transform*(0, tiffImg.height)
        ur_in = tiffImg.transform*(tiffImg.width, 0)

        ul_out = np.array(transform.transform(ul_in[0],ul_in[1]))
        lr_out = np.array(transform.transform(lr_in[0],lr_in[1]))
        ll_out = np.array(transform.transform(ll_in[0],ll_in[1]))
        ur_out = np.array(transform.transform(ur_in[0],ur_in[1]))


        bbox = np.array((ul_out,ur_out,lr_out,ll_out))

        UL = bbox.min(axis = 0)
        LR = bbox.max(axis = 0)

        rot_angle = angle_between(ur_out - ul_out, [1,0])*180/math.pi

        orig = [[0,0],
                [img_mat.shape[1],0],
                [0,img_mat.shape[0]],
                [img_mat.shape[1],img_mat.shape[0]]]

        img_mat_rot = rotate(img_mat,-rot_angle)

        rots = rot(img_mat,orig,-rot_angle)

        height,width,channels = img_mat_rot.shape
        imgSize = np.array([width,height])

        UTM_bounds = np.array([LR,UL])

        utm_range = LR - UL
        pixSizes = utm_range / imgSize

        transform_rot = Affine(pixSizes[0], 0, UL[0],
                               0, pixSizes[1], UL[1])

        contourPixel = utm_to_pix(imgSize, UTM_bounds.T, contourUTM)

        img_mat_rot = cv2.flip(img_mat_rot,0)

        for i in range(len(contourPixel)-1):
            cv2.line(img_mat_rot, tuple(contourPixel[i]), tuple(contourPixel[i+1]), (0,0,255), 2)

        #for i in range(len(rots)):
        #    cv2.circle(img_mat_rot,(int(rots[i][0]),int(rots[i][1])),8,(255,0,0),thickness=-1)

        img_mat_rot = cv2.flip(img_mat_rot,0)

    return img_mat_rot, UTM_bounds, transform_rot
