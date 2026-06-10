import numpy as np
import os
import sys
import argparse
from datetime import datetime
import utm
#import cv2
import math
import xml.etree.ElementTree as ET
from utils import *
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import rasterio as rio
import pandas as pd
import random
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import glob


def visualize(refUTM, xmlPaths, contourPath, bgImgPath, bgUTMPath, covThresh):

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
    contourUTM = np.load(contourPath, allow_pickle=True)
    contourPixel = utm_to_pix(imgSize, bgImgUTM, contourUTM)
    contourPolygon = Polygon(contourUTM)

    # Load reference coordinates (full coverage of surge region)
    refPixel = utm_to_pix(imgSize, bgImgUTM, refUTM)
    refPolygon = Polygon(refUTM)

    # Draw glacier contour onto background image
    for i in range(len(contourPixel)-1):
        cv2.line(bgImg, tuple(contourPixel[i]), tuple(contourPixel[i+1]), (0,0,255))

    # Draw reference polygon
    for i in range(len(refPixel)-1):
        cv2.line(bgImg, tuple(refPixel[i]), tuple(refPixel[i+1]), (0,0,255))

    # Load coordinaates of TIF image from xml file, translate to pixel locations
    coverages = []
    for n in range(len(xmlPaths)):
        xmlPath = xmlPaths[n]

        bboxLatlon = xml_to_latlon(xmlPath)
        bboxUTM = np.array([utm.from_latlon(i[0],i[1])[0:2] for i in bboxLatlon])
        bboxPixel = utm_to_pix(imgSize, bgImgUTM, bboxUTM)
        bboxPolygon = Polygon(bboxUTM)

        #print(bboxPolygon.intersects(refPolygon))
        if bboxPolygon.intersects(contourPolygon) and bboxPolygon.intersects(refPolygon):
            coverage = (refPolygon.intersection(bboxPolygon).area/refPolygon.area)*100

            if coverage > covThresh:
                coverages.append([xmlPath[7:-4],coverage])

                boundsUTM = np.concatenate([np.min(bboxUTM,axis=0), np.max(bboxUTM,axis=0)])

                UL = bboxUTM.min(axis = 0)
                LR = bboxUTM.max(axis = 0)
                UL_pix = utm_to_pix(imgSize, bgImgUTM, [UL])
                LR_pix = utm_to_pix(imgSize, bgImgUTM, [LR])

                # Draw TIF image  boundaries onto background image
                for i in range(len(bboxUTM)):
                    cv2.line(bgImg, tuple(bboxPixel[i]), tuple(bboxPixel[i+1]), (255,0,0))

                cv2.putText(bgImg, str(n), tuple(bboxPixel[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) )

    # Flip image to display in correct orientation
    bgImg = cv2.flip(bgImg,0)

    # Display image and wait for user input
    Image.fromarray(bgImg).show()

    return np.array(coverages)


#multi = ['Data/WV01_20121028211517/WV01_20121028211517_102001001FE5C400_12OCT28211517-P1BS-052903605040_01_P005.xml','Data/WV01_20110321212221/WV01_20110321212221_1020010011DD0F00_11MAR21212221-P1BS-052514087070_01_P001.xml','Data/WV01_20121028211436/WV01_20121028211436_102001001D7C9900_12OCT28211436-P1BS-052903544050_01_P005.xml']
#visualize(['Data/WV02_20160625170309/f.xml'],'Config/mlp_test_negri/negri_contour.npy', 'Negribreen_20170707_rgb_flip.png','negri_utm.npy')
multi = glob.glob('WV_xml/*.xml')
print('{} Images'.format(len(multi)))

covThresh = 85
ref = np.array([[581928.0743054642, 8732024.745110476], [582351.8848467076, 8722812.46476134], [593871.825922322, 8722500.184071539], [592793.0354537026, 8731985.710024253]])
print(ref)
#visualize(multi,'Config/mlp_test_bering/bering_contour.npy','xgoogle_bering.jpg','xgoogle_bering_utm.npy')
coverages = visualize(ref, multi, 'Config/mlp_test_negri/negri_contour.npy','Negribreen_20170707_rgb_flip.png','negri_utm.npy',covThresh)
print('{} Images meet coverage threshold of {}%'.format(coverages.shape[0], covThresh))
print(coverages[np.argsort(coverages[:,1])])
#visualize('Data/WV02_20160625170309/WV02_20160625170309.xml','Config/mlp_test_negri/negri_contour.npy','Negribreen_20170707_rgb_flip.png','negri_utm.npy','Data/WV02_20160625170309/WV02_20160625170309_(201,268)_split.npy')
