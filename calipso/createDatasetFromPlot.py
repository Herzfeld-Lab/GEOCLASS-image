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
imgPaths = cfg['img_path']
classEnum = cfg['class_enum']
density_path = cfg['density_path']
density_data = np.load(density_path, allow_pickle=True)
den = density_data[1]
density1 = den[0]
density2 = den[1]
density3 = den[2]
density4 = den[3]
density5 = den[4]
density6 = den[5]


def split_image(image_path, tile_width, tile_height, max, output_folder):
    # Open the image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_width, img_height = img.size

    # Prepare a list to store coordinates and tiles
    dataArray = []
    rows = img_height // tile_height
    cols = img_width // tile_width

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through the image to create tiles
    for row in range(rows):
        for col in range(cols):
            totDensity = []
            left, upper = col * tile_width, row * tile_height
            right, lower = left + tile_width, upper + tile_height
            splitImg_np = img_np[upper:lower, left:right]
            splitImg_width = splitImg_np.shape[0]
            splitImg_height = splitImg_np.shape[1]
            if splitImg_width != tile_width or splitImg_height != tile_height:
                print("Skipped image of shape", splitImg_np.shape)
                continue
            splitImg = scaleImage(splitImg_np, max)  # Assuming you have a scaling function

            # Ensure the image is RGB
            if splitImg.ndim == 2 or splitImg.shape[2] == 1:
                print('not rgb')
                splitImg = np.stack([splitImg] * 3, axis=-1)

            coords = [left, upper, -1, 0]
            totDensity.append(density1[upper:lower, left:right])
            totDensity.append(density2[upper:lower, left:right])
            totDensity.append(density3[upper:lower, left:right])
            totDensity.append(density4[upper:lower, left:right])
            totDensity.append(density5[upper:lower, left:right])
            totDensity.append(density6[upper:lower, left:right])

            
            dataArray.append(coords + [totDensity] + [splitImg])

            # Save the tile as an image
            tile_filename = f"tile_{row}_{col}.png"  # You can choose a different extension or format
            tile_filepath = os.path.join(output_folder, tile_filename)
            tile_img = Image.fromarray(splitImg)
            tile_img.save(tile_filepath)

    print(f'Created {len(dataArray)} split images and saved to {output_folder}\n')
    return dataArray

img = Image.open(imgPaths).convert("RGB")
imgnp = np.array(img)
max = get_img_sigma(imgnp)
output_folder = "split_images"
pix_coords_list = split_image(imgPaths, winSize[0], winSize[1], max, output_folder)

pix_coords_np = np.array(pix_coords_list, dtype='object')


info = {'filename': imgPaths,
        'winsize_pix': winSize,
        'class_enumeration': classEnum}

save_array_full = np.array([info, pix_coords_np], dtype='object')

dataset_path = "%d"%(pix_coords_np.shape[0]) + "_(%d,%d)_split"%(winSize[0],winSize[1]) + "_calipso"

cfg['npy_path'] = dataset_path+'.npy'

np.save(dataset_path, save_array_full)

print('Created {} total split images'.format(pix_coords_np.shape[0]))

print('Saved Full Dataset to {}\n'.format(dataset_path))

