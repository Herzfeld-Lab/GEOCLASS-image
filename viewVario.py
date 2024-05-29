

import os                                               # to set current working directory 
import numpy as np                                      # arrays and matrix math
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # plotting
import imageio.v2 as imageio
from utils import *
from numba import jit  # for numerical speed up

os.chdir("/home/twickler/Desktop/VarioCalculator")
filename = "1223991.tif"
img = imageio.imread(filename)
lagThresh = 0.8

# If numLag is greater than smallest image dimension * lagThresh, ovverride
imSize = img.shape
imRangeNS = imSize[0]*lagThresh
imRangeEW = imSize[1]*lagThresh
diagImSize = int(math.floor(np.sqrt((imSize[0]**2)+(imSize[1]**2))))
imRangeDiag = diagImSize*lagThresh
#Use of 3-4-5 rectangle
lagStepNS = 3
numLagNS = int(math.floor(imRangeNS / lagStepNS))
lagStepEW = 4
numLagEW = int(math.floor(imRangeEW / lagStepEW))
lagStepDiag = 5
numLagDiag = int(math.floor(imRangeDiag / lagStepDiag))
vario = np.zeros((4, max(numLagNS, numLagEW, numLagDiag)))
NSlag = []
EWlag = []
# For each value of lag, calculate directional variogram in given direction
for i,h in enumerate(range(1,numLagNS*lagStepNS,lagStepNS)):
    # North/South direction
    NSlag.append(h)
    diff = img[h:,:]-img[0:-h,:] 
    numPairs = diff.shape[0]*diff.shape[1]
    if numPairs != 0:
        v_h = (1. / numPairs) * np.sum(diff*diff)
        vario[0,i] = v_h
    #print("North/South Direction:")
    #print("Number of lag steps:", numLagNS)
    #print("Shape of diff:", diff.shape)
    #print("Number of pairs:", numPairs)


for i,h in enumerate(range(1,numLagEW*lagStepEW,lagStepEW)):
    # East/West direction
    EWlag.append(h)
    diff = img[:, :-h] - img[:, h:]
    numPairs = diff.shape[0]*diff.shape[1]
    if numPairs != 0:
        v_h = (1. / numPairs) * np.sum(diff*diff)
        vario[1,i] = v_h
    #print("East/West Direction:")
    #print("Number of lag steps:", numLagEW)
    #print("Shape of diff:", diff.shape)
    #print("Number of pairs:", numPairs)

# Diagonal direction (top right to bottom left)
for i, h in enumerate(range(1, numLagDiag * lagStepDiag, lagStepDiag)):
    # Calculate differences for diagonal direction (top right to bottom left)
    diff = img[NSlag[i]:, EWlag[i]:] - img[:-NSlag[i], :-EWlag[i]]
    if diff.shape[0]!=0 and diff.shape[1]!=0:
        numPairs = diff.shape[0] * diff.shape[1]
    elif diff.shape[0]!=0 and diff.shape[1] == 0:
        numPairs = diff.shape[0]
    elif diff.shape[1]!=0 and diff.shape[0] == 0:
        numPairs = diff.shape[1]
    if numPairs != 0:
        v_h = (1. / numPairs) * np.sum(diff * diff)
        vario[2, i] = v_h
    

# Diagonal direction (bottom right to top left)
for i, h in enumerate(range(1, numLagDiag * lagStepDiag, lagStepDiag)):
    # Calculate differences for diagonal direction (bottom right to top left)
    diff = img[:-NSlag[i], EWlag[i]:] - img[NSlag[i]:, :-EWlag[i]]
    if diff.shape[0] > 0 and diff.shape[1] > 0:
        numPairs = diff.shape[0] * diff.shape[1]
    if diff.shape[0]==0 or diff.shape[1]==0:
        if diff.shape[0]!=0 and diff.shape[1] == 0:
            numPairs = diff.shape[0]
        elif diff.shape[1]!=0 and diff.shape[0] == 0:
            numPairs = diff.shape[1]
    if numPairs != 0:
        v_h = (1. / numPairs) * np.sum(diff * diff)
        vario[3, i] = v_h
print(vario) 









"""
# Plot the variogram for North/South direction
plt.plot(range(1, numLagNS * lagStepNS, lagStepNS), vario[0], label='North/South')
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Variogram for North/South Direction')
plt.legend()
plt.show()

# Plot the variogram for East/West direction
plt.plot(range(1, numLagEW * lagStepEW, lagStepEW), vario[1], label='East/West')
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Variogram for East/West Direction')
plt.legend()
plt.show()

# Plot the variogram for diagonal direction (top right to bottom left)
plt.plot(range(1, numLagDiag * lagStepDiag, lagStepDiag), vario[2], label='Diagonal (top right to bottom left)')
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Variogram for Diagonal Direction (top right to bottom left)')
plt.legend()
plt.show()

# Plot the variogram for diagonal direction (bottom right to top left)
plt.plot(range(1, numLagDiag * lagStepDiag, lagStepDiag), vario[3], label='Diagonal (bottom right to top left)')
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Variogram for Diagonal Direction (bottom right to top left)')
plt.legend()
plt.show()
def calculate_semivariance(image, lag):
    nonzero_pixels = np.argwhere(image != 0)
    distances = pdist(nonzero_pixels, metric='euclidean')  # Pairwise distances between nonzero pixels
    semivariances = pdist(image_array[nonzero_pixels], metric='sqeuclidean')  # Pairwise semivariances
    filtered_pairs = [(distances[i], semivariances[i]) for i in range(len(distances)) if lag[0] <= distances[i] < lag[1]]
    distances_filtered, semivariances_filtered = zip(*filtered_pairs)
    return np.mean(semivariances_filtered)
"""