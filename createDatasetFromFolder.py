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
parser.add_argument("--check_imgs", type=bool, default=None)
parser.add_argument("--geotiff_num", type=int, default=None)
parser.add_argument("--class_num", type=int, default=None)
args = parser.parse_args()
#establishing check_imgs
if args.check_imgs:
    checkIMG = args.check_imgs
    tiffcheckNum = args.geotiff_num
    classcheckNum = args.class_num
# Read config file
with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Assign variables
pix_coords_list = []
dataset_path = cfg['npy_path']   
dataset = np.load(dataset_path, allow_pickle=True)
dataset_info = dataset[0]
transforms = dataset_info['transform']
img_path = cfg['training_img_path']
num_class = cfg['num_classes']
label_data = np.load(dataset_path, allow_pickle=True)
split_info_save = label_data[1]

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
        if (checkIMG == True) and (tiffcheckNum == int(tiffNum)) and (classcheckNum == int(n)):       
            for imgNum,imagePath in enumerate(imagePaths):
                if int(imgNum) == int(tiffNum):
                    print("testing image: ", index)
                    img = rio.open(imagePath)
                    imageMatrix = img.read(1)
                    max = get_img_sigma(imageMatrix[::10,::10])
                    splitImg_np = imageMatrix[x:x+winSize[0],y:y+winSize[1]]
                    splitImg_np = scaleImage(splitImg_np, max)
                    if (splitImg_np.shape[0] == 0) or (splitImg_np.shape[1] == 0):
                        print("Error with an image, class: ", n, "image source: ", tiffNum, "index: ", index)
        #Printing out image to check
        """
        geotiff = rio.open(dataset_info['filename'][int(tiffNum)])
        tiff_image_matrix = geotiff.read(1)
        tiff_image_max = get_img_sigma(tiff_image_matrix[::10,::10])
        img = tiff_image_matrix[x:x+win_size[0],y:y+win_size[1]]
        img = scaleImage(img, tiff_image_max)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        fileName = str(label)+str(index)+str(tiffNum)
        p = os.path.join(filePath, fileName)
        fp = p + '.tif'
        qimg.save(fp,"tif")
        """
        pix_coords_list.append([float(x),float(y),float(x_utm),float(y_utm),float(label),float(conf),float(tiffNum)])
        z+=1
        index = ''

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

#Saving .npy file
save_array_full = np.array([info, pix_coords_np], dtype='object')
dataset_path = args.config[:-7] + "_%d"%(num_class)+"_%d"%(z)

cfg['training_img_npy'] = dataset_path+'.npy'

np.save(dataset_path, save_array_full)

print('Saved Full Dataset to {}\n'.format(dataset_path))

f = open(args.config, 'w')
f.write(generate_config_silas(cfg))
f.close()
