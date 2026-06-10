import sys
sys.path.append('/home/twickler/ws/GEOCLASS-image/NN_Class')
from utils import *
from auto_rotate_geotiff import *
#CST20240312
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QVBoxLayout, QCheckBox, QSlider, QLineEdit, QPushButton, QButtonGroup, QMainWindow, QGridLayout
#CST 20240308
from PyQt5.QtGui import QPixmap, QImage, QFont, QGuiApplication, QFont, qRgb

from PyQt5.QtCore import Qt

from PIL.ImageQt import ImageQt
import os
import shutil


from Models import *
from Dataset import *

import rasterio as rio

from scipy.spatial import KDTree

from matplotlib.pyplot import get_cmap
from matplotlib.colors import ListedColormap

import yaml


class SplitImageTool(QWidget):
    def __init__(self, cfg_path, checkpoint=None, netcdf=False):
        """Initialize the Split Image GUI Tool."""
        super().__init__()
        print('-------- Initializing App --------')

        # Load configuration file
        self.cfg_path = cfg_path
        with open(self.cfg_path, 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        # Initialize dataset and UI
        self.initDataset()
        self.initUI()
        self.initBgImages()
        self.getNewImages(self.image_indices)  # Load first three images
        self.setLayout(self.master_layout)
        self.show()

    ### **📌 PART 1: Dataset Initialization** ###
    def initDataset(self):
        """Loads dataset properties and initializes required attributes."""
        print("-------- Loading Dataset --------")
        self.label_path = self.cfg['npy_path']
        self.label_data = np.load(self.label_path, allow_pickle=True)
        self.split_info = self.label_data[1][:, :5]  # Extract necessary columns

        self.dataset_info = self.label_data[0]
        self.class_enum = self.dataset_info['class_enumeration']
        self.win_size = self.dataset_info['winsize_pix']

        self.image_indices = [0, 1, 2]  # Initial indices for the three images

    ### **📌 PART 2: UI Initialization** ###
    def initUI(self):
        """Initializes UI components with a grid layout for images."""
        print("-------- Setting Up UI --------")
        self.master_layout = QVBoxLayout()
        self.image_grid = QGridLayout()

        # Create labels for three background images and small split images
        self.bg_labels = [QLabel(self) for _ in range(3)]
        self.small_labels = [QLabel(self) for _ in range(3)]

        for i in range(3):
            self.bg_labels[i].setAlignment(Qt.AlignCenter)
            self.small_labels[i].setAlignment(Qt.AlignCenter)

            self.image_grid.addWidget(self.bg_labels[i], 0, i)  # Background images
            self.image_grid.addWidget(self.small_labels[i], 1, i)  # Small split images

        self.master_layout.addLayout(self.image_grid)

    ### **📌 PART 3: Load Background Images** ###
    def initBgImages(self):
        """Loads and displays three background images."""
        print("-------- Loading Background Images --------")
        self.bg_images = []
        self.bg_pixmaps = []

        for i in range(3):
            if i >= len(self.dataset_info['filename']):
                break  # Avoid out-of-range errors

            # Load background image
            plot = Image.open(self.dataset_info['filename'][i])
            tiff_matrix = np.array(plot)

            # Scale down for display
            scale_factor = max(1, int(tiff_matrix.shape[0] / 75))  # Prevent division by zero
            bg_img_scaled = tiff_matrix[::scale_factor, ::scale_factor]

            # Convert to QImage & QPixmap
            qimage = QImage(bg_img_scaled.data, bg_img_scaled.shape[1], bg_img_scaled.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaledToWidth(300)  # Adjust width as needed

            self.bg_images.append(bg_img_scaled)
            self.bg_pixmaps.append(pixmap)

            # Update the QLabel widgets
            self.bg_labels[i].setPixmap(pixmap)

    ### **📌 PART 4: Load and Display Small Split Images** ###
    def getNewImages(self, indices):
        """Loads and displays three small images based on the given indices."""
        print("-------- Loading Small Split Images --------")
        for i in range(3):
            if i >= len(indices) or indices[i] >= len(self.split_info):
                continue  # Skip if out of range

            x, y, label, conf, _ = self.split_info[indices[i]]
            x, y = int(x), int(y)

            # Extract small image
            img = self.bg_images[i][y:y + self.win_size[1], x:x + self.win_size[0]]
            img = img[:, :, :3] if img.shape[2] == 4 else img  # Handle RGBA images
            img = img.astype(np.uint8)

            # Convert to QPixmap
            qimage = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaledToWidth(100)  # Adjust size as needed

            # Display small images
            self.small_labels[i].setPixmap(pixmap)

    ### **📌 PART 5: Image Navigation System** ###
    def updateImages(self, direction):
        """Updates displayed images when navigating with A/D keys."""
        if direction == "left":
            self.image_indices = [max(0, i - 1) for i in self.image_indices]
        elif direction == "right":
            self.image_indices = [min(len(self.split_info) - 1, i + 1) for i in self.image_indices]

        self.getNewImages(self.image_indices)

    def keyPressEvent(self, event):
        """Handles keyboard input for navigation."""
        if event.key() == Qt.Key_A:
            self.updateImages("left")
        elif event.key() == Qt.Key_D:
            self.updateImages("right")

### **📌 PART 6: Run the Application** ###
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--load_labels", type=str, default=None)
    parser.add_argument("--netcdf", action="store_true")
    args = parser.parse_args()

    # Read config file
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    app = QApplication(sys.argv)
    ex = SplitImageTool(args.config, args.load_labels, args.netcdf)
    sys.exit(app.exec_())