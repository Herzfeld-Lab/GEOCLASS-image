import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
from PIL.ImageQt import ImageQt
from utils import *
from Models import *
from Dataset import *
import rasterio as rio
import numpy as np
import utm
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import pyproj
from pyproj import Transformer
from pyproj import CRS
from auto_rotate_geotiff import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import yaml

class SplitImageTool(QWidget):

    def __init__(self, cfg_path, checkpoint=None):
        super().__init__()

        # Initialize GUI Window properties
        print('-------- Initializing App --------')
        screen_resolution = app.desktop().screenGeometry()
        self.title = 'Split Image Labeling tool'
        self.width, self.height = screen_resolution.width(), screen_resolution.height()
        self.setGeometry(0, 0, self.width, self.height)
        self.setWindowTitle(self.title)

        # Load Tiff Image and split image data
        print('-------- Loading Tiff Image --------')
        self.cfg_path = cfg_path
        with open(self.cfg_path, 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile)

        if checkpoint:
            self.label_path = checkpoint
        else:
            self.label_path = self.cfg['txt_path']

        data = np.load(self.label_path, allow_pickle=True)

        # Initialize dataset properties
        print('-------- Initializing Dataset --------')
        self.split_info = data[1]
        self.dataset_info = data[0]
        self.geotiff = rio.open(self.dataset_info['filename'])
        self.tiff_image_matrix = self.geotiff.read(1)
        self.tiff_image_max = self.tiff_image_matrix.max()
        self.class_enum = self.dataset_info['class_enumeration']
        self.utm_epsg_code = self.cfg['utm_epsg_code']
        self.win_size = self.dataset_info['winsize_pix']
        #self.contour_np = np.load(self.dataset_info['contour_path'])
        self.contour_np = np.load(self.cfg['contour_path'])
        self.contour_polygon = Polygon(self.contour_np)
        self.lookup_tree = KDTree(self.split_info[:,2:4])

        # Declare dimensions for GUI components
        print('-------- Initializing GUI --------')
        self.split_image_pos = (int(self.width/4) - int(self.win_size[1]/2), 10)
        self.button_grid_pos = (0,0)
        self.tiff_image_pos = (int(self.width/2), 10)
        self.split_image_class_pos = ((int(self.width/4) - int(self.win_size[1]/2), self.win_size[0] + 20))
        self.conf_slider_pos = (self.split_image_class_pos[0], self.split_image_class_pos[1] + 20)
        self.buttons_pos = (10, self.conf_slider_pos[1] + 20)
        self.visualize_button_pos = (self.conf_slider_pos[0], self.conf_slider_pos[1] + 20)

        # --- Initialize UI elements ---
        # Current split Image container
        self.split_image_pixmap = QPixmap()
        self.split_image_label = QLabel(self)
        self.split_image_label.setMargin(0)
        self.split_image_label.setAlignment(Qt.AlignCenter)
        #self.split_image_label.move(self.split_image_pos[0], self.split_image_pos[1])

        # Background Image container
        self.tiff_image_pixmap = QPixmap()
        self.tiff_image_label = QLabel(self)
        self.tiff_image_label.setMargin(0)
        #self.tiff_image_label.setAlignment(Qt.AlignCenter)
        #self.tiff_image_label.move(self.tiff_image_pos[0], self.tiff_image_pos[1])

        # Current class label container
        self.split_image_class = QLabel(self)
        self.split_image_class.setMargin(0)
        self.split_image_class.setAlignment(Qt.AlignCenter)
        self.split_image_class.setFont(QFont("Helvetica", 15, QFont.Bold))

        self.split_image_conf = QLabel(self)
        self.split_image_conf.setMargin(0)
        self.split_image_conf.setAlignment(Qt.AlignCenter)
        self.split_image_conf.setFont(QFont("Helvetica", 15, QFont.Bold))

        # Buttons
        self.button_containers = []

        self.visualize_button = QCheckBox('Visualize Labels', self)
        self.visualize_button.stateChanged.connect(self.visualize_callback)

        self.new_class_field = QLineEdit()
        self.new_class_field.textChanged.connect(self.textChangedCallback)
        self.new_class_field.returnPressed.connect(self.newClass)
        self.new_class_label = ''

        self.new_class_button = QPushButton('New Class')
        self.new_class_button.clicked.connect(self.newClassCallback)

        # Slider
        self.conf_thresh_slider = QSlider(Qt.Horizontal)
        self.conf_thresh_slider.setFocusPolicy(Qt.StrongFocus)
        self.conf_thresh_slider.setTickPosition(QSlider.TicksBothSides)
        self.conf_thresh_slider.setTickInterval(10)
        self.conf_thresh_slider.setSingleStep(1)
        self.conf_thresh_slider.setMinimum(0)
        self.conf_thresh_slider.setMaximum(99)
        self.conf_thresh_slider.setValue(0)
        self.conf_thresh_slider.valueChanged.connect(self.slider_callback)

        # Initialize Image and Dimensions container variables
        self.bg_img_scaled = None
        self.bg_img_cv = None
        self.bg_qimage = None
        self.bg_img_utm = None
        self.image_index = None
        self.clicked = None
        self.cmap = plt.get_cmap('tab20')
        self.conf_thresh = 0
        self.selected_classes = np.ones(len(self.class_enum))

        self.initBgImage()
        self.getNewImage(0)
        self.initUI()
        self.setLayout(self.master_layout)
        self.show()

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

    def initUI(self):

        self.master_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.visualization_widgets = QVBoxLayout()
        self.class_buttons = QVBoxLayout()
        self.split_text = QHBoxLayout()

        self.initClassButtons()

        self.split_text.addSpacing(200)
        self.split_text.addWidget(self.split_image_class)
        self.split_text.addWidget(self.split_image_conf)
        self.split_text.addSpacing(200)

        self.visualization_widgets.addWidget(self.split_image_label)
        self.visualization_widgets.addLayout(self.split_text)
        self.visualization_widgets.addWidget(self.conf_thresh_slider)
        self.visualization_widgets.addWidget(self.visualize_button)

        self.left_layout.addLayout(self.visualization_widgets)
        self.left_layout.addLayout(self.class_buttons)

        self.master_layout.addLayout(self.left_layout)
        self.master_layout.addWidget(self.tiff_image_label)

    def addClassButton(self, i, className):
        buttonContainer = QHBoxLayout()

        labelButton = QPushButton('%d: %s'%(i, className), self)
        #labelButton.setAlignment(Qt.AlignLeft)
        labelButton.clicked.connect(self.makeClassLabelCallbacks(i))

        labelToggle = QCheckBox(self)
        labelToggle.toggle()
        labelToggle.stateChanged.connect(self.makeClassToggleCallbacks(i))

        buttonContainer.addSpacing(200)
        buttonContainer.addWidget(labelToggle)
        buttonContainer.addWidget(labelButton)
        buttonContainer.addSpacing(200)
        self.class_buttons.addLayout(buttonContainer)

    def initClassButtons(self):

        self.new_class_layout = QHBoxLayout()
        self.new_class_layout.addSpacing(200)
        self.new_class_layout.addWidget(self.new_class_field)
        self.new_class_layout.addSpacing(200)
        self.class_buttons.addLayout(self.new_class_layout)

        for i, className in enumerate(self.class_enum):
            self.addClassButton(i, className)


    def initBgImage(self, visualize=False):

        # Scale down tiff image for visualization and convert to 8-bit grayscale
        self.bg_img_scaled = self.tiff_image_matrix[::25,::25]
        self.bg_img_scaled = self.bg_img_scaled / self.bg_img_scaled.max()
        self.bg_img_scaled = (self.bg_img_scaled * 255).astype('uint8')

        self.bg_img_scaled = cv2.cvtColor(self.bg_img_scaled,cv2.COLOR_GRAY2RGB)

        if visualize:
            for splitImg in self.split_info:

                if splitImg[5] > self.conf_thresh and self.selected_classes[int(splitImg[4])]:

                    x,y = int(splitImg[0]/25),int(splitImg[1]/25)
                    ul = (y,x)
                    lr = (y+7,x+7)

                    c = (np.array(self.cmap(int(splitImg[4]))[:3])*255).astype('int')

                    cv2.rectangle(self.bg_img_scaled, ul,lr,(int(c[0]),int(c[1]),int(c[2])),thickness=-1)


        # Rotate tiff to align North and plot glacier contour
        self.bg_img_cv, self.bg_img_utm, self.bg_img_transform = auto_rotate_geotiff(self.geotiff, self.bg_img_scaled, self.utm_epsg_code, self.contour_np)
        height,width,_ = self.bg_img_scaled.shape

        # Convert to QImage from cv and wrap in QPixmap container
        self.bg_qimg = QImage(self.bg_img_cv.data,self.bg_img_cv.shape[1],self.bg_img_cv.shape[0],self.bg_img_cv.shape[1]*3,QImage.Format_RGB888)
        self.tiff_image_pixmap = QPixmap(self.bg_qimg)
        self.tiff_image_pixmap = self.tiff_image_pixmap.scaledToWidth(int(self.width/2) - 10)

        # Get scaling factor between cv and q image
        bg_img_cv_size = np.array(self.bg_img_cv.shape[:-1])
        bg_img_q_size = np.array((self.tiff_image_pixmap.size().height(), self.tiff_image_pixmap.size().width()))

        self.scale_factor = bg_img_cv_size / bg_img_q_size
        self.tiff_image_label.setPixmap(self.tiff_image_pixmap)

    def scaleImage(self, img):
        img = img/self.tiff_image_max
        img = (img*255).astype('uint8')
        return img

    def getNewImage(self, index):
         self.image_index = index

         # Grab info of split image at index
         x,y,x_utm,y_utm,label,conf = self.split_info[index]
         x,y,x_utm,y_utm,label = int(x),int(y),int(x_utm),int(y_utm),int(label)

         # Get split image from image matrix
         img = self.tiff_image_matrix[x:x+self.win_size[0],y:y+self.win_size[1]]
         img = self.scaleImage(img)
         img = Image.fromarray(img).convert("L")
         img = ImageQt(img)

         # Wrap split image in QPixmap
         self.split_image_pixmap = QPixmap.fromImage(img)
         self.split_image_label.setPixmap(self.split_image_pixmap)

         # Update label text
         class_text = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
         if label == -1:
             class_text = "No class assigned yet"
             class_conf = ''
         else:
             class_text = "Class %d: %s"%(label, self.class_enum[label])
             class_conf = "Conf: {:.2%}".format(conf)

         self.split_image_class.setText(class_text)
         self.split_image_conf.setText(class_conf)

         bg_img = self.bg_img_cv.copy()
         height,width,_ = self.bg_img_cv.shape
         imgSize = np.array([width,height])
         pix_coords = utm_to_pix(imgSize, self.bg_img_utm.T, np.array([[x_utm,y_utm]]))

         # Draw crosshairs
         cv2.circle(bg_img,(pix_coords[0][0],height-pix_coords[0][1]),8,(255,0,0),thickness=-1)
         cv2.line(bg_img, (pix_coords[0][0], 0), (pix_coords[0][0], height), (255,0,0),thickness=3)
         cv2.line(bg_img, (0, height-pix_coords[0][1]), (width, height-pix_coords[0][1]), (255,0,0),thickness=3)

         height,width,channels = bg_img.shape
         bg_img = QImage(bg_img.data,width,height,width*channels,QImage.Format_RGB888)
         background_image = QPixmap(bg_img)
         background_image = background_image.scaledToWidth(int(self.width/2) - 10)
         self.bg_img_scaled = background_image
         self.tiff_image_label.setPixmap(background_image)

    def label(self, class_label):
        self.split_info[self.image_index][4] = class_label
        self.split_info[self.image_index][5] = 1
        self.getNewImage(self.image_index)
        self.update()

    def mousePressEvent(self, QMouseEvent):
        click_pos = QMouseEvent.pos()
        click_pos_scaled = self.tiff_image_label.mapFromGlobal(click_pos)
        click_pos_scaled = np.array([click_pos_scaled.y(), click_pos_scaled.x()]) * self.scale_factor
        click_pos_utm = self.bg_img_transform * (click_pos_scaled[1], self.bg_img_cv.shape[0] - click_pos_scaled[0])

        if Point(click_pos_utm[0], click_pos_utm[1]).within(self.contour_polygon):
            d,i = self.lookup_tree.query(click_pos_utm)
            self.getNewImage(i)

    def keyPressEvent(self, event):
        index = self.image_index

        if event.key() <= 57 and event.key() >= 48:
            key_map = {48: 0,
                       49: 1,
                       50: 2,
                       51: 3,
                       52: 4,
                       53: 5,
                       54: 6,
                       55: 7,
                       56: 8,
                       57: 9}
            self.label(key_map[event.key()])

        elif event.key() == 65: #Left arrow key
            index -= 1
        elif event.key() == 68: #Right arrow key
            index += 1

        self.getNewImage(index)
        self.update()

    def makeClassLabelCallbacks(self, classLabel):
        def class_label_callback():
            self.label(classLabel)
        return class_label_callback

    def makeClassToggleCallbacks(self, classLabel):
        def class_toggle_callback():
            self.selected_classes[classLabel] = not self.selected_classes[classLabel]
        return class_toggle_callback

    @pyqtSlot()
    def on_click(self):
        self.update()

    @pyqtSlot()
    def new_class(self):
        self.update()

    def visualize_callback(self, state):
        if state == Qt.Checked:
            self.initBgImage(visualize=True)
        else:
            self.initBgImage(visualize=False)

    def slider_callback(self):
        value = self.conf_thresh_slider.value()
        self.conf_thresh = value/100.0

    def textChangedCallback(self, text):
        self.new_class_label = text

    def newClass(self):
        self.class_enum.append(self.new_class_label)
        self.selected_classes = np.ones(len(self.class_enum))
        self.addClassButton(len(self.class_enum)-1, self.new_class_label)
        self.update()

    def newClassCallback(self):
        print('')

    def closeEvent(self, event):
        save_array = np.array([self.dataset_info, self.split_info])
        np.save(self.label_path, save_array)
        print('DONE')
        event.accept()
        return

if __name__ == '__main__':

    # Parse command line flags
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    #ex = SplitImageTool('Data/classes_10/Worldview_Image/WV02_20160625170309_1030010059AA3500_16JUN25170309-P1BS-500807681050_01_P004_u16ns3413_(201,268)_dataset.npy')
    #ex = SplitImageTool('Output/28-01-2020_18:20/labels/labeled_epoch_624.npy')
    ex = SplitImageTool(args.config, args.load_checkpoint)

    sys.exit(app.exec_())
