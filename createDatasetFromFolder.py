import os
import sys
import argparse
import yaml
import utm
from utils import *
from Split_Image_Explorer_PyQt5 import *

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

pix_coords_list = []
dataset_path = cfg['npy_path']   
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
transforms = dataset_info['transform']
img_path = cfg['training_img_path']
num_class = cfg['num_classes']
label_data = np.load(dataset_path, allow_pickle=True)
split_info_save = label_data[1]
split_info = split_info_save[split_info_save[:,6] == 0]
index = ''
z=0
for x in range(0,num_class):
    Dir = str(img_path+"/"+str(x))
    startIndex = len(Dir)+1 +len(str(x))
    imgs = getImgPaths(Dir)
    for img in imgs:
        for y in range(startIndex, (len(img)-5)):
            index += str(img[y])
        tiffNum = img[-5]
        x,y,x_utm,y_utm,label,conf,_ = split_info[int(index)]
        label = x
        x,y,x_utm,y_utm,label = int(x),int(y),int(x_utm),int(y_utm),int(label)
        index = ''
        ##Follow steps in create from geotiff to write out a npy using this information
        pix_coords_list.append([x,y,x_utm,y_utm,label,conf,tiffNum])
        z+=1

topDir = cfg['img_path']
winSize = cfg['split_img_size']
contourPath = cfg['contour_path']
epsgCode = cfg['utm_epsg_code']
classEnum = cfg['class_enum']
geotiff_bool = True
imgPaths = getImgPaths(topDir)

print('**** Saving Dataset ****')
pix_coords_np = np.array(pix_coords_list)

info = {'filename': imgPaths,
        'is_geotiff': geotiff_bool,
        'contour_path': contourPath,
        'winsize_pix': winSize,
        #'winsize_utm': UTM_winSize,
        'transform': transforms,
        'class_enumeration': classEnum}


save_array_full = np.array([info, pix_coords_np], dtype='object')
dataset_path = args.config[:-7] + "_%d"%(num_class)+"_%d"%(z)

cfg['training_img_npy'] = dataset_path+'.npy'

np.save(dataset_path, save_array_full)

print('Saved Full Dataset to {}\n'.format(dataset_path))

f = open(args.config, 'w')
f.write(generate_config_silas(cfg))
f.close()
