

import os                                               # to set current working directory 
import numpy as np                                      # arrays and matrix math
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # plotting
import imageio.v2 as imageio
import math
import cv2
from numba import jit  # for numerical speed up

def getVarioSilas(img, lagThresh = 0.8):
    imSize = img.shape
    imRangeNS = imSize[0]*lagThresh
    imRangeEW = imSize[1]*lagThresh
    diagImSize = int(math.floor(np.sqrt((imSize[0]**2)+(imSize[1]**2))))
    imRangeDiag = diagImSize*lagThresh
    if imSize[0] < imSize[1]:
        #Use of 3-4-5 rectangle
        lagStepNS = 3
        numLagNS = int(math.floor(imRangeNS / lagStepNS))
        lagStepEW = 4
        numLagEW = int(math.floor(imRangeEW / lagStepEW))
        lagStepDiag = 5
        numLagDiag = int(math.floor(imRangeDiag / lagStepDiag))

    elif imSize[0] > imSize[1]:
        #Use of 3-4-5 rectangle
        lagStepNS = 4
        numLagNS = int(math.floor(imRangeNS / lagStepNS))
        lagStepEW = 3
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
        #Use NSlag and EWlag, so it goes 3-4 in the correct dimensions, essentially giving a lagstep of 5
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

    
    # Plot the variogram for North/South direction
    plt.plot(range(1, numLagNS * lagStepNS, lagStepNS), vario[0], label='North/South')
    plt.plot(range(1, numLagEW * lagStepEW, lagStepEW), vario[1], label='East/West')
    plt.plot(range(1, numLagDiag * lagStepDiag, lagStepDiag), vario[2], label='Diagonal (top right to bottom left)')
    plt.plot(range(1, numLagDiag * lagStepDiag, lagStepDiag), vario[3], label='Diagonal (bottom right to top left)')
    plt.xlabel('Lag Distance')
    plt.ylabel('Semivariance')
    plt.title('Variogram for Crevasse')
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


    return vario



filename = "/home/twickler/paper1figures/crev.png"
img = imageio.imread(filename)
lagThresh = 0.8

# If numLag is greater than smallest image dimension * lagThresh, ovverride
normalVar = getVarioSilas(img,lagThresh)



#This section was used to determine how many unique variogram combinations one would get by rotating and flipping an image
"""
#horizontal filp
horizontalFlip = cv2.flip(img, 0)
horizontalFlipVar = getVario(horizontalFlip,lagThresh)
#vertical flip
verticalFlip = cv2.flip(img,1)
verticalFlipVar = getVario(verticalFlip,lagThresh)
#roate 90 clockwise
rotate90 = np.rot90(img, k=1)
rotate90Var = getVario(rotate90, lagThresh)
#roate 180 clockwise
rotate180 = np.rot90(img, k=2)
rotate180Var = getVario(rotate180, lagThresh)
#roate 270 clockwise
rotate270 = np.rot90(img, k=3)
rotate270Var = getVario(rotate270, lagThresh)

horz90 = np.rot90(horizontalFlip, k=1)
horz90Var = getVario(horz90, lagThresh)
horz180 = np.rot90(horizontalFlip, k=2)
horz180Var = getVario(horz180, lagThresh)
horz270 = np.rot90(horizontalFlip, k=3)
horz270Var = getVario(horz270, lagThresh)


vert90 = np.rot90(verticalFlip, k=1)
vert90Var = getVario(vert90, lagThresh)
vert180 = np.rot90(verticalFlip, k=2)
vert180Var = getVario(vert180, lagThresh)
vert270 = np.rot90(verticalFlip, k=3)
vert270Var = getVario(vert270, lagThresh)

both = cv2.flip(horizontalFlip,1)
bothVar = getVario(both, lagThresh)
both90 = np.rot90(both, k=1)
both90Var =getVario(both90, lagThresh)
both180 = np.rot90(both, k=1)
both180Var =getVario(both180, lagThresh)
both270 = np.rot90(both, k=1)
both270Var =getVario(both270, lagThresh)
"""

"""
#Test if statements to see what variograms are equal, replace the first argument in all the if statements with any variogram
#to see what other variograms are equal
if np.array_equal(vert90Var,verticalFlipVar):
    print("Normal, vert flip")
if np.array_equal(vert90Var,horizontalFlipVar):
    print("Normal, horz flip")
if np.array_equal(vert90Var,rotate90Var):
    print("Normal, 90 rotate")
if np.array_equal(vert90Var,rotate180Var):
    print("Normal, 180 rotate")
if np.array_equal(vert90Var,rotate270Var):
    print("Normal, 270 rotate")
if np.array_equal(vert90Var,vert90Var):
    print("Normal, vert 90 rotate")
if np.array_equal(vert90Var,vert180Var):
    print("Normal, vert 180 rotate")
if np.array_equal(vert90Var,vert270Var):
    print("Normal, vert 270 rotate")
if np.array_equal(vert90Var,horz90Var):
    print("Normal, horz 90 rotate")
if np.array_equal(vert90Var,horz180Var):
    print("Normal, horz 180 rotate")
if np.array_equal(vert90Var,horz270Var):
    print("Normal, horz 270 rotate")
if np.array_equal(vert90Var,bothVar):
    print("Normal, both rotate")
if np.array_equal(vert90Var,both90Var):
    print("Normal, both 90 rotate")
if np.array_equal(vert90Var,both180Var):
    print("Normal, both 180 rotate")
if np.array_equal(vert90Var,both270Var):
    print("Normal, both 270 rotate")
"""







