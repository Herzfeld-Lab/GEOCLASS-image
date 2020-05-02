#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  brew install pyqt5

if [[ "$OSTYPE" == "linux-gnu"]]; then
  sudo apt install python3-pyqt5

sudo pip3 install affine pillow opencv-python rasterio torch torchvision shapely sklearn utm pyproj matplotlib pandas
