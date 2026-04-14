import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore
from PIL.ImageQt import ImageQt
from utils_MS import *
from Models import *
from Dataset_MS import *
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

@njit
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
"""
def find_contour(img_mat):
    imgray = cv2.cvtColor(img_mat, cv2.COLOR_RGB2GRAY)
    imgray[imgray[:] > 0] = 255
    kernel = np.ones((5,5),np.uint8)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)
    imgray = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0].squeeze()
"""
def find_contour(img_mat):
    if img_mat.ndim == 2:
        # Already grayscale (panchromatic)
        imgray = img_mat.copy()
    elif img_mat.ndim == 3:
        # Multispectral (more than 3 bands)
        if img_mat.shape[2] > 3:
            # Use first band (most stable for mask)
            imgray = img_mat[:, :, 0]
        else:
            imgray = cv2.cvtColor(img_mat, cv2.COLOR_RGB2GRAY)

    else:
        raise ValueError("Unsupported image dimensions")
    #Mask
    imgray = imgray.copy()
    imgray[imgray > 0] = 255
    imgray = imgray.astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)
    for _ in range(2):
        imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)
        imgray = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No contours found in image.")

    return contours[0].squeeze()

def find_valid_bbox_pixels(img_mat):
    """
    Returns bounding box of non-zero image region in pixel coordinates.
    Works for panchromatic and multispectral imagery.
    """

    if img_mat.ndim == 2:
        mask = img_mat != 0
    elif img_mat.ndim == 3:
        mask = np.any(img_mat != 0, axis=2)
    else:
        raise ValueError("Unsupported image dimensions")

    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        raise ValueError("No valid image data found after rotation.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return x_min, x_max, y_min, y_max

def rotate_and_crop_geotiff(tiffInfo, tiffImg, img_mat, epsg_code, contourUTM, tiff_num):
    if img_mat.ndim == 3 and img_mat.shape[0] < 10:
        img_mat = np.moveaxis(img_mat, 0, -1)
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


        img_mat_rot = rotate(
            img_mat,
            -rot_angle,
            reshape=True,
            order=0,
            mode='constant',
            cval=0
        )

        rots = rot(img_mat,orig,-rot_angle)

        img_mat_rot = img_mat

        height,width,channels = img_mat_rot.shape
        imgSize = np.array([height,width])

        UTM_bounds = np.array([LR,UL])

        utm_range = LR - UL
        pixSizes = utm_range / imgSize

        transform_rot_1 = Affine(pixSizes[0], 0, transform.c,
                               0, -pixSizes[1], transform.f)

        print(transform)
        print(transform_rot_1)

        contourPixel = utm_to_pix(imgSize, UTM_bounds.T, contourUTM)

        img_mat_rot = cv2.flip(img_mat_rot,0)


        for i in range(len(contourPixel)-1):
            cv2.line(img_mat_rot, tuple(contourPixel[i]), tuple(contourPixel[i+1]), (0,0,255), 2)

        #for i in range(len(rots)):
        #    cv2.circle(img_mat_rot,(int(rots[i][0]),int(rots[i][1])),8,(255,0,0),thickness=-1)

        img_mat_rot = cv2.flip(img_mat_rot,0)

        print(img_mat_rot.shape)

        #return img_mat, bbox, transform

    else:
        # Get transform from tiff image georeferencing data to UTM
        bbox = get_geotiff_bounds(tiffImg, epsg_code)
        ul_out,ur_out = bbox[0],bbox[1]

        # Get UTM boundaries
        UL = bbox.min(axis = 0)
        LR = bbox.max(axis = 0)

        # Find angle between image orientation and due north
        rot_angle = angle_between(ur_out - ul_out, [1,0])*180/math.pi

        # Rotate image to line up with north/south
        img_mat_rot = rotate(
            img_mat,
            -rot_angle,
            reshape=True,
            order=0,
            mode='constant',
            cval=0
        )

        # Get pixel -> UTM affine transform for rotated image
        height,width,channels = img_mat_rot.shape
        imgSize = np.array([width,height])
        UTM_bounds = np.array([LR,UL])
        utm_range = LR - UL
        pixSizes = utm_range / imgSize
        transform_rot = Affine(pixSizes[0], 0, UL[0],
                               0, pixSizes[1], UL[1])

        # Find boundaries of actual image data (non-black pixels)
        height, width = img_mat_rot.shape[:2]

        x_min, x_max, y_min, y_max = find_valid_bbox_pixels(img_mat_rot)

        # Define pixel-space bbox corners
        pixel_bbox = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])

        # Convert bbox corners to UTM
        data_bbox_utm = []
        for px in pixel_bbox:
            data_bbox_utm.append(transform_rot * px)

        data_bbox_utm = np.array(data_bbox_utm)

        bbox_polygon = Polygon(data_bbox_utm)

        # Find intersection of data boundary contour and glacier contour
        bbox_polygon = Polygon(data_bbox_utm)
        contour_polygon = Polygon(contourUTM)
        intersection = bbox_polygon.intersection(contour_polygon)
        if intersection.geom_type == 'MultiPolygon':
             # Access individual polygons within the MultiPolygon
            polygons = list(intersection.geoms)
   	    # Find the polygon with the maximum area
            intersection_poly = max(polygons, key=lambda a: a.area)            
            newpoly = np.array(list(intersection_poly.exterior.coords))
        else:
            intersection_poly = intersection
            newpoly = np.array(list(intersection_poly.exterior.coords))

        # Get bounding box in UTM of intersection
        cropped_bbox = intersection_poly.bounds

        UL = np.array([cropped_bbox[0],cropped_bbox[1]]).astype('int')
        LR = np.array([cropped_bbox[2],cropped_bbox[3]]).astype('int')

        bbox_pix = utm_to_pix(imgSize, UTM_bounds.T, np.array([UL,LR]))

        # Crop image to intersection boundaries
        img_mat_rot = img_mat_rot[height-bbox_pix[1][1]:height-bbox_pix[0][1],bbox_pix[0][0]:bbox_pix[1][0],:]
        height,width,channels = img_mat_rot.shape
        imgSize = np.array([width,height])

        # Get affine transform for cropped image
        UTM_bounds = np.array([LR,UL])
        utm_range = UTM_bounds[0] - UTM_bounds[1]

        pixSizes = utm_range / imgSize
        transform_rot_1 = Affine(pixSizes[0], 0, UL[0],
                               0, pixSizes[1], UL[1])

        # Plot glacier contour onto cropped image and return
        contourPixel = utm_to_pix(imgSize, UTM_bounds.T, contourUTM)

        img_mat_rot = cv2.flip(img_mat_rot,0)

        for i in range(len(contourPixel)-1):
        
            cv2.line(img_mat_rot, tuple(contourPixel[i]), tuple(contourPixel[i+1]), (0,0,255), 2)


        img_mat_rot = cv2.flip(img_mat_rot,0)


    return img_mat_rot, UTM_bounds, transform_rot_1
