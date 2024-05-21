"""
This code was taken from https://github.com/GeostatsGuy/PythonNumericalDemos/blob/master/GeostatsPy_variogram_from_image.ipynb and modified
Michael Pyrcz, Associate Professor, University of Texas at Austin
"""
import geostatspy.GSLIB as GSLIB                       # GSLIB utilies, visualization and wrapper
import geostatspy.geostats as geostats                 # variogram calculations  
import os                                               # to set current working directory 
import numpy as np                                      # arrays and matrix math
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # plotting
import imageio.v2 as imageio
from utils import *
from numba import jit  # for numerical speed up
os.chdir("/home/twickler/Desktop/VarioCalculator")
filename = "1225990.tif"
img = imageio.imread(filename)
numLag = 14
vario = directional_vario(img, numLag, lagThresh = 0.8)
print(vario)

"""


ny = imageBW.shape[0]
nx = imageBW.shape[1]


count = 0
print('Image size, ny = ' + str(ny) + ' , nx = ' + str(nx))


nlagx,variox,nppx = geostats.gam(imageBW,tmin=-9999,tmax=9999,xsiz=1.0,ysiz=1.0,ixd=2,iyd=0,nlag=350,isill=1.0)
nlagy,varioy,nppy = geostats.gam(imageBW,tmin=-9999,tmax=9999,xsiz=1.0,ysiz=1.0,ixd=0,iyd=2,nlag=150,isill=1.0)

plt.subplot(211)
plt.imshow(imageBW,cmap=plt.cm.Greys)
plt.xlabel('X (pixels)'); plt.ylabel('Y (pixels)'); plt.title('Image for Variogram Analysis')

plt.subplot(212)
plt.scatter(nlagx,variox,s=20,color='orange',edgecolor='black',label='X')
plt.scatter(nlagy,varioy,s=20,color='red',edgecolor='black',label='Y')
plt.plot([0,250],[1,1],color='black')
plt.xlabel(r'Lag Distance, $\bf{h}$ (pixels)'); plt.ylabel(r'Variogram $\gamma$($\bf{h}$)'); plt.title('Directional Variograms')
plt.xlim([0,250]); plt.ylim([0,1.1]); plt.grid(); plt.legend(loc='lower right')

plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.6, wspace=0.2, hspace=0.3); plt.show()
print("Done")
"""