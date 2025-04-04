import sys
sys.path.append('/home/twickler/ws/GEOCLASS-image/NN_Class')
from utils import *
from PIL import Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import rasterio as rio
from rasterio.transform import Affine

import yaml

import warnings
import tarfile
import re
import geopandas as gpd

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)   
warnings.filterwarnings('error')

# Load config parameters
winSize = cfg['split_img_size']
#imgPaths = cfg['img_path']
classEnum = cfg['class_enum']
#multiple paths

density = []
tab = []
asr = []

density_path = cfg['density_path']
density_data = np.loadtxt(density_path, dtype= float)
density = density_data

tab_paths = cfg['tab_path']
for tab_path in tab_paths:
    tab_data = np.loadtxt(tab_path, dtype= float)
    tab.append(tab_data)
asr_paths = cfg['asr_path']
for asr_path in asr_paths:
    asr_data = np.loadtxt(asr_path, dtype= float)
    asr.append(asr_data)


"""
 #single path
density_paths = cfg['density_path']
density = np.loadtxt(density_paths, dtype= float)
tab_paths = cfg['tab_path']
tab = np.loadtxt(tab_paths, dtype= float)
asr_paths = cfg['asr_path']
asr = np.loadtxt(asr_paths, dtype= float)
"""

def largest(arr):
    mx = arr[0]
    n = len(arr)
    for i in range(1, n):
        if arr[i] > mx:
            mx = arr[i]

    return mx

def split_image(den, tab, asr, tile_width, tile_height, output_folder, lat, lon):
    # Open the image and convert to RGB
    #Replace this with something that will create the image that will be used for classification
    #set -9999 to 0 just for visualization (less than 0)
    #plot log of density?
    #rgb or grayscale? use density for all channels
    if len(den) <= 3:
        
        max = 0 #only will work if density conatins more than one entry
        for dens in den[0]:
            for each_den in dens:
                if each_den > max:
                    max = each_den
                if each_den < 0:
                    each_den = 0
        img = den[0]
    else:
        max = 0
        for dens in den:
            for each_den in dens:
                if each_den > max:
                    max = each_den
                if each_den < 0:
                    each_den = 0
  
        
        img = den
    #image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
    #img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.shape
    img_np = np.array(img)
    #print(img_np.shape)
    #img padding for edge cases
    pad_height = (tile_height - img_height % tile_height) % tile_height
    pad_width = (tile_width - img_width % tile_width) % tile_width
    img_padded = np.pad(img_np, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    new_height, new_width = img_padded.shape[:2]   

    """
    padded_tab = np.pad(
    tab,
    ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
    mode='constant',
    constant_values=0  # Fill with 0s (can change)
    )   
    padded_asr = np.pad(
    asr,
    ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
    mode='constant',
    constant_values=0  # Fill with 0s (can change)
    )   

    padded_density = np.pad(
    den,
    ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
    mode='constant',
    constant_values=0  # Fill with 0s (can change)
    )   

    """
    #Multiple paths
    """
    density1 = den[0]
    density2 = den[1]
    density3 = den[2]

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

    padded_density1 = np.pad(
    density1,
    ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
    mode='constant',
    constant_values=0  # Fill with 0s (can change)
    )   
    padded_density2 = np.pad(
    density2,
    ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
    mode='constant',
    constant_values=0  # Fill with 0s (can change)
    )   
    padded_density3 = np.pad(
    density3,
    ((0, pad_height), (0, pad_width)),  # Pad only bottom & right
    mode='constant',
    constant_values=0  # Fill with 0s (can change)
    )   
    """
  
    # Prepare a list to store coordinates and tiles
    dataArray = []
    rows = new_height // tile_height
    cols = new_width // tile_width
    #print(rows,cols)
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through the image to create tiles
    for row in range(rows):
        for col in range(cols):
            """
            totDensity = []
            totTab = []
            totAsr = []
            lat_lon = []
            """
            left, upper = col * tile_width, row * tile_height
            right, lower = left + tile_width, upper + tile_height
            splitImg_np = img_padded[upper:lower, left:right]
            splitImg_width = splitImg_np.shape[0]
            splitImg_height = splitImg_np.shape[1]
            if splitImg_width != tile_width or splitImg_height != tile_height:
                print("Skipped image of shape", splitImg_np.shape)
                continue
            splitImg = (scaleImage(splitImg_np, max))/255.0  # Assuming you have a scaling function
            splitImg = np.stack([splitImg*0.5, splitImg*0.75, splitImg], axis=-1)
            splitImg = (splitImg*255.0).astype(np.uint8)
            # Ensure the image is RGB
            """
            if splitImg.ndim == 2 or splitImg.shape[2] == 1:
                print('not rgb')
                splitImg = np.stack([splitImg] * 3, axis=-1)
            """
            coords = [left, upper, -1, 0]
            """
            totDensity = padded_density[upper:lower, left:right]
            totTab = padded_tab[upper:lower, left:right]
            totAsr = padded_asr[upper:lower, left:right]
            """

            #Multiple paths
            """
            totDensity.append(padded_density1[upper:lower, left:right])
            totDensity.append(padded_density2[upper:lower, left:right])
            totDensity.append(padded_density3[upper:lower, left:right])
            totTab.append(padded_tab1[upper:lower, left:right])
            totTab.append(padded_tab2[upper:lower, left:right])
            totTab.append(padded_tab3[upper:lower, left:right])
            totAsr.append(padded_asr1[upper:lower, left:right])
            totAsr.append(padded_asr2[upper:lower, left:right])
            totAsr.append(padded_asr3[upper:lower, left:right])
            """
            if lat or lon == None:
                lat_lon = 0
            else:
                lat_lon.append(lat[upper:lower, left:right])
                lat_lon.append(lon[upper:lower, left:right])
            

            dataArray.append(coords + [splitImg] + [lat_lon])
            #dataArray.append(coords + [splitImg] + [totDensity] + [totTab] + [totAsr] + [lat_lon])

            # Save the tile as an image
            tile_filename = f"tile_{row}_{col}.png"  # You can choose a different extension or format
            tile_filepath = os.path.join(output_folder, tile_filename)
            tile_img = Image.fromarray(splitImg)
            tile_img.save(tile_filepath)

    print(f'Created {len(dataArray)} split images and saved to {output_folder}\n')
    return dataArray

#img = Image.open(imgPaths).convert("RGB")
#imgnp = np.array(img)
#max = get_img_sigma(imgnp) #wont need this 
output_folder = "split_images"
pix_coords_list = split_image(density, tab, asr, winSize[0], winSize[1], output_folder, lat=None, lon=None)

pix_coords_np = np.array(pix_coords_list, dtype='object')

max = 0
density1 = tab[0]
density2 = tab[1]
density3 = tab[2]
for dens in density1:
    for each_den in dens:
        if each_den > max:
            max = each_den
        if each_den < 0:
            each_den = 0
        
        img1 = density1[::-1, :]
for dens in density2:
    for each_den in dens:
        if each_den > max:
            max = each_den
        if each_den < 0:
            each_den = 0
        
        img2 = density2[::-1, :]
for dens in density3:
    for each_den in dens:
        if each_den > max:
            max = each_den
        if each_den < 0:
            each_den = 0
        
        img3 = density3[::-1, :]


img1_np = np.array(img1)
img2_np = np.array(img2)
img3_np = np.array(img3)
fullImg1 = (scaleImage(img1_np, max))/255.0  # Assuming you have a scaling function
fullImg2 = (scaleImage(img2_np, max))/255.0 
fullImg3 = (scaleImage(img3_np, max))/255.0 
fullImg1 = np.stack([fullImg1*0.5,fullImg1*0.75,fullImg1], axis=-1)
fullImg1 = (fullImg1*255.0).astype(np.uint8)
fullImg2 = np.stack([fullImg2*0.5,fullImg2*0.75,fullImg2], axis=-1)
fullImg2 = (fullImg2*255.0).astype(np.uint8)
fullImg3 = np.stack([fullImg3*0.5,fullImg3*0.75,fullImg3], axis=-1)
fullImg3 = (fullImg3*255.0).astype(np.uint8)
img_filename1 = f"calipso_1_{density.shape[0]}_{density.shape[1]}.png"  # You can choose a different extension or format
imgPaths1 = img_filename1
img_filename2 = f"calipso_2_{density.shape[0]}_{density.shape[1]}.png"  # You can choose a different extension or format
imgPaths2 = img_filename2
img_filename3 = f"calipso_3_{density.shape[0]}_{density.shape[1]}.png"  # You can choose a different extension or format
imgPaths3 = img_filename3
full_img1 = Image.fromarray(fullImg1)
full_img1.save(imgPaths1)
full_img2 = Image.fromarray(fullImg2)
full_img2.save(imgPaths2)
full_img3 = Image.fromarray(fullImg3)
full_img3.save(imgPaths3)

info = {'filename': [imgPaths1, imgPaths2, imgPaths3],
        'winsize_pix': winSize,
        'class_enumeration': classEnum}

save_array_full = np.array([info, pix_coords_np], dtype='object')

dataset_path = "%d"%(pix_coords_np.shape[0]) + "_(%d,%d)_split"%(winSize[0],winSize[1]) + "_calipso"

cfg['npy_path'] = dataset_path+'.npy'

np.save(dataset_path, save_array_full)

print('Created {} total split images'.format(pix_coords_np.shape[0]))

print('Saved Full Dataset to {}\n'.format(dataset_path))

