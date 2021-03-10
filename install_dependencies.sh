#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  sudo pip3 install pyqt5
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
  sudo apt install python3-pyqt5
else
  echo "System not recognized - Please install on MacOS or Ubuntu Linux"
fi

sudo pip3 install affine pillow opencv-python-headless rasterio torch torchvision shapely sklearn utm pyproj matplotlib pandas pyyaml scikit-image
