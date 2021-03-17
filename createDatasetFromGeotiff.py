from utils import *

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import rasterio as rio
from rasterio.transform import Affine

import yaml

import warnings
import tarfile
import re
import geopandas as gpd

'''
# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
warnings.filterwarnings('error')
'''


def create_dataset(cfg):

    # Load config parameters
    topDir = cfg['img_path']
    winSize = cfg['split_img_size']
    contourPath = cfg['contour_path']
    epsgCode = cfg['utm_epsg_code']
    classEnum = cfg['class_enum']

    pix_coords = np.zeros((0,7))
    transforms = []
    imgPaths = getImgPaths(topDir)
    geotiff_bool = []
    print("\n{} Tiff Images to split".format(len(imgPaths)))

    for IMG_NUM,imgPath in enumerate(imgPaths):

        # Load tiff image
        print("**** Loading Tiff Image {}: {} ****".format(IMG_NUM, imgPath))
        try:
            geoTiff = rio.open(imgPath)
            isGeotiff = True
        except rio.errors.NotGeoreferencedWarning as warning:
            print('NOTE: tiff image has no georeferencing data. Attempting to get transform from metadata tarfile'.format(imgPath))
            warnings.filterwarnings("ignore")
            geoTiff = rio.open(imgPath)
            isGeotiff = False

        geotiff_bool.append(isGeotiff)

        # Load UTM glacier contour
        print("**** Loading Contour File ****")
        if contourPath != 'None':
            try:
                contourUTM = np.load(contourPath)
            except FileNotFoundError as e:
                print('ERROR: {} not found.')
                exit(1)
        else:
            print('No Contour File Specified. See \'Generating a Contour File\' in the README')
            contourUTM = get_geotiff_bounds(geoTiff,epsgCode)

        contourPolygon = Polygon(contourUTM)

        print('**** Splitting Image ****')
        pix_coords_img = split_geotiff(geoTiff, winSize, epsgCode, contourPolygon)
        print('Created {} split images\n'.format(len(pix_coords_img)))
        pix_coords_img = np.append(pix_coords_img, np.array([[IMG_NUM]*len(pix_coords_img)]).T, axis=1)
        pix_coords = np.append(pix_coords, pix_coords_img, axis=0)

        '''
        band1 = geoTiff.read(1)
        imgSize = band1.shape
        print('Image size: {}x{}'.format(imgSize[0],imgSize[1]))
        '''

        # Get geo transform for tiff image
        if not isGeotiff:
            with tarfile.open(imgPath[:-4]+'.tar') as tf:
                pattern = re.compile('.*_PIXEL_SHAPE.*')
                contents = tf.getnames()
                for filename in contents:
                    if pattern.search(filename):
                        shapefile = filename
                        tf.extract(filename, path=topDir)
                gdf = gpd.read_file('/'.join([topDir,shapefile[:-4]+'.shp']))
                #print(gdf['geometry'][0])
                        #crs_in = CRS.from_wkt(f.read().decode('utf-8'))

            crs_in = CRS.from_epsg(4326)
            crs_utm = CRS.from_epsg(epsgCode)
            transform = Transformer.from_crs(crs_in, crs_utm)
        else:
            crs_in = CRS.from_wkt(geoTiff.crs.wkt)
            crs_utm = CRS.from_epsg(epsgCode)
            transform = Transformer.from_crs(crs_in, crs_utm)

        # Calculate image bounding box in UTM
        if not isGeotiff:
            xmlPath = imgPath[:-4] + '.xml'

            try:
                bboxLatlon, GSD = xml_to_latlon(xmlPath)
            except FileNotFoundError as e:
                print('ERROR: {} not found. Please include a .xml metadata file'.format(xmlPath))
                exit(1)

            bboxUTM = [0]*len(bboxLatlon)
            for i in range(len(bboxLatlon)):
                bboxUTM[i] = transform.transform(bboxLatlon[i][0], bboxLatlon[i][1])
            bboxUTM = np.array(bboxUTM)

            print('**** Calculating Image Transform ****')

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

            y_shear = (LR_img[1] - LR_max[1]) / (UR_max[0] - UL_max[0])

            x_shear = (LL_img[0] - LL_min[0]) / (LL_max[1] - UL_max[1])

            utm_affine_transform = Affine(pixSizes[0], 0, UL_min[0],
                                   0, pixSizes[1], UL_min[1])

            utm_affine_transform = utm_affine_transform * Affine.shear(rot_angle_y, rot_angle_x)

            ULmaybe = utm_affine_transform * (0,0)
            LRmaybe = utm_affine_transform * (imgSize[0], imgSize[1])

            utm_affine_transform = Affine(GSD[0], 0, bboxUTM[0][0],
                                            0, -GSD[1], bboxUTM[0][1])

        else:
            bboxUTM = get_geotiff_bounds_utm(geoTiff, epsgCode)

        '''
        print('IMAGE TRANSFORM:')
        if not isGeotiff:
            print(utm_affine_transform)
        else:
            print(geoTiff.transform)
        '''

        # Perform split image. Get all split images within UTM glacier contour
        # print('**** Splitting Image ****')
        '''
        count = 0
        for row,i in enumerate(range(0,imgSize[0],winSize[0])):

            for col,j in enumerate(range(0,imgSize[1],winSize[1])):

                UL_pix = (i,j)
                LR_pix = (i+winSize[0],j+winSize[1])

                if not isGeotiff:
                    UL_UTM = np.asarray(utm_affine_transform*(UL_pix[1],imgSize[0] - UL_pix[0]))
                    LR_UTM = np.asarray(utm_affine_transform*(LR_pix[1],imgSize[0] - LR_pix[0]))

                else:
                    ul_in = geoTiff.transform*(UL_pix[1],UL_pix[0])
                    lr_in = geoTiff.transform*(LR_pix[1],LR_pix[0])

                    UL_UTM = np.array(transform.transform(ul_in[0],ul_in[1]))
                    LR_UTM = np.array(transform.transform(lr_in[0],lr_in[1]))

                splitImg = band1[i:i+winSize[0],j:j+winSize[1]]

                if splitImg.shape != tuple(winSize):
                    continue

                if Point(UL_UTM[0],UL_UTM[1]).within(contourPolygon) and Point(LR_UTM[0],LR_UTM[1]).within(contourPolygon):
                    if splitImg[splitImg==0].shape[0] == 0:
                        pix_coords_list.append([i,j,UL_UTM[0],UL_UTM[1],-1,0,IMG_NUM])
                        count += 1
        print('Created {} split images\n'.format(count))
        '''

        if not isGeotiff:
            transforms.append(utm_affine_transform)
        else:
            transforms.append(None)

    print('**** Saving Dataset ****')
    #pix_coords_np = np.array(pix_coords_list)
    pix_coords_np = pix_coords

    info = {'filename': imgPaths,
            'contour_path': contourPath,
            'winsize_pix': winSize,
            #'winsize_utm': UTM_winSize,
            'transform': transforms,
            'class_enumeration': classEnum}

    save_array_full = np.array([info, pix_coords_np], dtype='object')

    dataset_path = args.config[:-7] + "_%d"%(pix_coords_np.shape[0]) + "_(%d,%d)_split"%(winSize[0],winSize[1])

    cfg['npy_path'] = dataset_path+'.npy'

    np.save(dataset_path, save_array_full)

    print('Created {} total split images'.format(pix_coords_np.shape[0]))

    print('Saved Full Dataset to {}\n'.format(dataset_path))

    f = open(args.config, 'w')
    f.write(generate_config(cfg))
    f.close()


def split_geotiff(geoTiff, winSize, epsgCode, contourPolygon):
    band1 = geoTiff.read(1)
    imgSize = band1.shape

    # Get transformer between input CRS and output CRS (UTM)
    crs_in = CRS.from_wkt(geoTiff.crs.wkt)
    crs_UTM = CRS.from_epsg(epsgCode)
    transformer = Transformer.from_crs(crs_in, crs_UTM)

    pix_coords_np = np.zeros((0,6))

    for row,i in enumerate(range(0,imgSize[0],winSize[0])):
        for col,j in enumerate(range(0,imgSize[1],winSize[1])):

            UL_pix = (i,j)
            LR_pix = (i+winSize[0],j+winSize[1])

            pixCoords = np.array([UL_pix, LR_pix])
            UTMCoords = pix_to_utm(pixCoords, geoTiff, transformer)
            UL_UTM,LR_UTM = UTMCoords[0], UTMCoords[1]

            splitImg = band1[i:i+winSize[0],j:j+winSize[1]]

            if splitImg.shape != tuple(winSize):
                continue

            if Point(UL_UTM[0],UL_UTM[1]).within(contourPolygon) and Point(LR_UTM[0],LR_UTM[1]).within(contourPolygon):
                if splitImg[splitImg==0].shape[0] == 0:
                    pix_coords_np = np.append(pix_coords_np, [[i,j,UL_UTM[0],UL_UTM[1],-1,0]], axis=0)

    return pix_coords_np


if __name__ == '__main__':

    # Parse command line flags
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    # Read config file
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    warnings.filterwarnings('error')

    create_dataset(cfg)
