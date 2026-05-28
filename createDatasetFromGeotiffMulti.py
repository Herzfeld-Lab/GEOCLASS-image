from utils_MS import *

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import rasterio as rio
from rasterio.transform import Affine

import yaml
from pathlib import Path

import warnings
import tarfile
import re
import geopandas as gpd


def is_bad(tile, nodata_value=None):
    """
    Flag tiles with effectively no texture or fill values.
    Handles both 2D (PAN) and 3D (MS) arrays.
    """
    # Check for low texture/variation
    if tile.ndim == 3:
        # For multispectral (3D), check each band separately
        stds = [np.nanstd(band) for band in tile]
        if all(std < 1e-6 for std in stds):
            return True
    elif tile.ndim == 2:
        # For panchromatic (2D), standard check
        if np.nanstd(tile) < 1e-6:
            return True

    # Check for tiles that are all nodata or uniform fill
    if nodata_value is not None:
        if np.all(tile == nodata_value):
            return True
    else:
        # Fallback when nodata metadata is missing
        # Only reject if entire tile is uniform (very strict condition)
        if tile.ndim == 3:
            # For 3D, check if all bands are completely uniform
            uniform = all(np.all(band == 0) or np.all(band == 65535)
                         for band in tile)
            if uniform:
                return True
        else:
            # For 2D, check if entire tile is uniform
            if np.all(tile == 0) or np.all(tile == 65535):
                return True

    return False


# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
warnings.filterwarnings('error')

# Load config parameters
topDir = cfg['img_path']
winSize = cfg['split_img_size']
contourPath = cfg['contour_path']
epsgCode = cfg['utm_epsg_code']
classEnumMS = cfg['class_enum_MS']
classEnumPAN = cfg['class_enum_PAN']
convFactor = cfg['conversion_factor']
msWinSize = [0,0]

msWinSize[0] = int(winSize[0] / convFactor)
msWinSize[1] = int(winSize[1] / convFactor)

pix_coords_list = []
pan_pix_coords_list = []
ms_pix_coords_list = []
transforms = []
imgPaths = Path(topDir)
geotiff_bool = []
trueImgPaths = []
PATH_NUM = -1
IMG_NUM = -1


scene_folders = [f for f in imgPaths.iterdir() if f.is_dir()]


print("\n{} Scenes to split".format(len(scene_folders)))


#Have a folder for each scene

for folder in scene_folders:
    PATH_NUM += 1
    multiSpectral = []
    panSpectral = []
    i = 0
    ms_data = None
    pan_data = None
    msImg = None
    panImg = None
    ms_path = None
    pan_path = None
    ms_isGeotiff = None
    pan_isGeotiff = None
    ms_utm_affine_transform = None
    pan_utm_affine_transform = None
    ms_transformer = None
    pan_transformer = None
    ms_nodata = None
    pan_nodata = None

    tif_paths = sorted(folder.glob("*.tif"), key=lambda p: ("P1BS" in p.name, p.name))
    for imgPath in tif_paths:
        # Load tiff image
        print(imgPath)
        print("**** Loading Tiff Image {}: {} ****".format(IMG_NUM+1, imgPath))
        try:
            tiffImg = rio.open(imgPath)
            isGeotiff = True
        except rio.errors.NotGeoreferencedWarning as warning:
            print('NOTE: tiff image has no georeferencing data. Attempting to get transform from metadata tarfile'.format(imgPath))
            warnings.filterwarnings("ignore")
            tiffImg = rio.open(imgPath)
            isGeotiff = False
           
        if tiffImg.count > 1:  
            msImg = tiffImg
            msPath = imgPath
            isMultiSpectral = True
            ms_data  = msImg.read()
            band1 = tiffImg.read()
            MSimg_h = ms_data.shape[1]
            MSimg_w = ms_data.shape[2]
            imgSize = ms_data[0].shape
            ms_nodata = tiffImg.nodata
            if ms_nodata is None:
                print(f"WARNING: MS image has no nodata value set: {imgPath}")
        else:
            panImg = tiffImg
            panPath = imgPath
            isMultiSpectral = False
            pan_data = panImg.read(1)
            band1 = tiffImg.read(1)
            imgSize = pan_data.shape
            h_pan, w_pan = pan_data.shape
            pan_nodata = tiffImg.nodata
            if pan_nodata is None:
                print(f"WARNING: PAN image has no nodata value set: {imgPath}")
        utm_affine_transform = None
        
        print('Image size: {}x{}'.format(imgSize[0],imgSize[1]))
        #Treat image as a non Geotiff if it has no crs data.
        if tiffImg.crs is not None:
            isGeotiff = True
        else:
            isGeotiff = False
            print("Not a GEOTIFF")
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
            #CST06182024
        else:
            if tiffImg.crs is not None:
                crs_in = CRS.from_wkt(tiffImg.crs.wkt)
            else:
                crs_in = CRS.from_epsg(4326)
            crs_utm = CRS.from_epsg(epsgCode)
            transform = Transformer.from_crs(crs_in, crs_utm)

        # Load UTM glacier contour
        print("**** Loading Glacier Contour ****")
        if contourPath != 'None':
            try:
                contourUTM = np.load(contourPath)
            except FileNotFoundError as e:
                print('ERROR: {} not found. Please include a .npy file with UTM coordinates of glacier boundary contour')
                exit(1)
        else:
            contourUTM = get_geotiff_bounds(tiffImg,epsgCode)
        contourPolygon = Polygon(contourUTM)
        #print(crs_in.to_wkt())
        #print('CRS EPSG code: {}'.format(crs_in.to_epsg()))

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

            utm_affine_transform = Affine(GSD, 0, bboxUTM[0][0],
                                            0, -GSD, bboxUTM[0][1])

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
        '''
        print('IMAGE TRANSFORM:')
        if not isGeotiff:
            print(utm_affine_transform)
        else:
            print(tiffImg.transform)
        '''

        if isMultiSpectral:
            ms_path = imgPath
            ms_isGeotiff = isGeotiff
            ms_utm_affine_transform = utm_affine_transform
            ms_transformer = transform
        else:
            pan_path = imgPath
            pan_isGeotiff = isGeotiff
            pan_utm_affine_transform = utm_affine_transform
            pan_transformer = transform

        # Perform split image. Get all split images within UTM glacier contour
    if ms_data is None or pan_data is None or ms_path is None or pan_path is None:
        print(f"Skipping {folder} (missing MS or PAN image)")
        continue

    # Use pan metadata for split geometry
    tiffImg = panImg
    isGeotiff = pan_isGeotiff
    utm_affine_transform = pan_utm_affine_transform
    transform = pan_transformer
    imgSize = pan_data.shape
    h_pan, w_pan = pan_data.shape

    IMG_NUM = len(trueImgPaths)
    print('**** Splitting Image ****')
    count = 0
    pair_coords = []

    for i in range(0, h_pan - winSize[0] + 1, winSize[0]):
        for j in range(0, w_pan - winSize[1] + 1, winSize[1]):

            UL_pix = (i,j)
            LR_pix = (i+winSize[0],j+winSize[1])

            ms_i = int(round(i / convFactor))
            ms_j = int(round(j / convFactor))
            ms_i_end = ms_i + msWinSize[0]
            ms_j_end = ms_j + msWinSize[1]

            if (ms_i < 0 or ms_j < 0 or
                ms_i_end > ms_data.shape[1] or
                ms_j_end > ms_data.shape[2]):
                continue

            pan_split = pan_data[i:i+winSize[0], j:j+winSize[1]]
            ms_split  = ms_data[:, ms_i:ms_i+msWinSize[0], ms_j:ms_j+msWinSize[1]]
            
            if is_bad(pan_split, pan_nodata) or is_bad(ms_split, ms_nodata):
                continue
                
            if not isGeotiff:
                UL_UTM = np.asarray(utm_affine_transform*(UL_pix[1],imgSize[0] - UL_pix[0]))
                LR_UTM = np.asarray(utm_affine_transform*(LR_pix[1],imgSize[0] - LR_pix[0]))

            else:
                ul_in = tiffImg.transform*(UL_pix[1],UL_pix[0])
                lr_in = tiffImg.transform*(LR_pix[1],LR_pix[0])

                UL_UTM = np.array(transform.transform(ul_in[0],ul_in[1]))
                LR_UTM = np.array(transform.transform(lr_in[0],lr_in[1]))

            #splitImg = band1[i:i+winSize[0],j:j+winSize[1]]

            if pan_split.shape != tuple(winSize):
                continue

            # pix_coords_list data is used to access actual data at runtime - avoids loading giant data all at once
            if Point(UL_UTM[0], UL_UTM[1]).within(contourPolygon) and Point(LR_UTM[0], LR_UTM[1]).within(contourPolygon):

                # Check for too-high fraction of nodata/fill pixels using the actual nodata value
                pan_nodata_pct = np.mean(pan_split == pan_nodata) if pan_nodata is not None else 0
                ms_nodata_pct = np.mean(ms_split == ms_nodata) if ms_nodata is not None else 0
                
                pan_valid = pan_nodata_pct < 0.5
                ms_valid = ms_nodata_pct < 0.5
                
                #print(pan_valid, "pan valid", pan_nodata_pct)
                #print(ms_valid, "ms valid", ms_nodata_pct)
                
                if pan_valid and ms_valid:
                    # Format: [pan_x, pan_y, ms_x, ms_y, utm_x, utm_y, pan_label, pan_conf, ms_label, ms_conf, img_num]
                    pair_coords.append([i, j, ms_i, ms_j, UL_UTM[0], UL_UTM[1], -1, 0, -1, 0, IMG_NUM])
                    #pair_coords.append([i, j, ms_i, ms_j, UL_UTM[0], UL_UTM[1], -1, 0, IMG_NUM])
                    count += 1
            

    if count > 0:
        pix_coords_list.extend(pair_coords)
        trueImgPaths.extend([ms_path, pan_path])
        geotiff_bool.extend([ms_isGeotiff, pan_isGeotiff])
        transforms.extend([
            ms_utm_affine_transform if not ms_isGeotiff else None,
            pan_utm_affine_transform if not pan_isGeotiff else None,
        ])
        print('Created {} split images\n'.format(count))
    else:
        print('Skipping pair (no valid split images)\n'.format(count))
    

print('**** Saving Dataset ****')
pix_coords_np = np.array(pix_coords_list)

info = {'filename': trueImgPaths,
        'is_geotiff': geotiff_bool,
        'contour_path': contourPath,
        'winsize_pix': winSize,
        'MS_winsize_pix': msWinSize,
        #'winsize_utm': UTM_winSize,
        'transform': transforms,
        'class_enumeration_MS': classEnumMS,
        'class_enumeration_PAN': classEnumPAN}

#print('DATASET INFO: ')
#print(json.dumps(info, indent=2))

save_array_full = np.array([info, pix_coords_np], dtype='object')

dataset_path = args.config[:-7] + "_%d"%(pix_coords_np.shape[0]) + "_(%d,%d)_split"%(winSize[0],winSize[1])

cfg['npy_path'] = dataset_path+'.npy'

np.save(dataset_path, save_array_full)

print('Created {} total split images'.format(pix_coords_np.shape[0]))

print('Saved Full Dataset to {}\n'.format(dataset_path))

f = open(args.config, 'w')
f.write(generate_config_silas(cfg))
f.close()
