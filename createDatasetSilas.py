from utils import *
import os
import sys
import argparse
import yaml
import utm
from utils import *
from Split_Image_Explorer import *

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("--check_imgs", type=bool, default=None)
parser.add_argument("--geotiff_num", type=int, default=None)
parser.add_argument("--class_num", type=int, default=None)
args = parser.parse_args()
#establishing check_imgs

# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Assign variables
dataset_path = cfg['npy_path']   
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
transforms = dataset_info['transform']
img_path = cfg['training_img_path']
num_class = cfg['num_classes']
label_data = np.load(dataset_path, allow_pickle=True)
split_info_save = label_data[1]
var = []
labels = []
index = ''
z=0

#Trying to print out images
#win_size = dataset_info['winsize_pix']
#filePath = 'TestImages'

#This loop is needed to track what class each image is
for n in range(0,num_class):
    Dir = str(img_path+"/"+str(n))
    startIndex = len(Dir)+1 +len(str(n))
    imgs = getImgPaths(Dir)
    #print(n)
    #Going through each image and finding the index and tiffNum
    for img in imgs:
        for y in range(startIndex, (len(img)-5)):
            index += str(img[y])
        tiffNum = img[-5]
        #print(index)
        #print('tiff num', tiffNum)
        split_info = split_info_save[split_info_save[:,6] == int(tiffNum)]
        x,y,x_utm,y_utm,label,conf,_ = split_info[int(index)]
        x,y,x_utm,y_utm,label = int(x),int(y),int(x_utm),int(y_utm),int(label)
        label = n
        conf = 0
        #Making sure the image is valid
        imgPath = cfg['img_path']
        imagePaths = getImgPaths(imgPath)
        winSize = dataset_info['winsize_pix']
        var.append(silas_directional_vario(img))
        labels.append(label)
        z+=1
        index = ''

print('**** Saving Dataset ****')
var_np = np.array(var)

#Saving .npy file
save_array_full = np.array([labels, var_np], dtype='object')
dataset_path = args.config[:-7] + "_VarNet_%d"%(num_class)+"_%d"%(z)

np.save(dataset_path, save_array_full)

print('Saved Full Dataset to {}\n'.format(dataset_path))

