#!/bin/bash


# Check if virtualenv is installed
if ! python3 -c "import venv" &> /dev/null
then
   echo "venv module is not available. Aborting."
   exit 1
fi


# Create a directory for the virtual environment
cd
cd Desktop


# Create the virtual environment
python3 -m venv venv
cd ~/venv
# Activate the virtual environment
source venv/bin/activate


# Provide instructions to the user
echo "Virtual environment created successfully. You can activate it by running:"
echo "source Desktop/venv/bin/activate"




if [[ "$OSTYPE" == "darwin"* ]]; then
 #pip3 install pyqt5
 pip3 install torch
 pip3 install rasterio
 pip3 install Pyyaml
 pip3 install pyqt5
 pip3 install pyqt6
 pip3 install pillow
 pip3 install scikit-learn
 pip3 install geopandas
 pip3 install torchvision
 pip3 install pandas
 pip3 install scikit-image
 pip3 install numba
 pip3 install opencv-python
 pip3 install netCDF4
 pip3 install utm
 pip3 install matplotlib
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
 ##apt install python3-pyqt5
 apt install python3-torch
 apt install python3-rasterio
 apt install python3-Pyyaml
 apt install python3-pyqt5
 apt install python3-pyqt6
 apt install python3-pillow
 apt install python3-scikit-learn
 apt install python3-geopandas
 apt install python3-torchvision
 apt install python3-pandas
 apt install python3-scikit-image
 apt install python3-numba
 apt install python3-opencv-python
 apt install python3-netCDF4
 apt install python3-utm
 apt install python3-matplotlib
else
 #echo "System not recognized - Please install on MacOS or Ubuntu Linux"

fi


pip3 install -r requirements.txt

