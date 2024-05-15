from utils import *
from auto_rotate_geotiff import *
#CST20240312
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QVBoxLayout, QCheckBox, QSlider, QLineEdit, QPushButton, QButtonGroup
#CST 20240308
from PyQt5.QtGui import QPixmap, QImage, QFont, QGuiApplication, QFont

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
    global numTiff
    numTiff = 0
    def __init__(self, cfg_path, checkpoint=None, netcdf=False):
        super().__init__()

        # Initialize GUI Window properties
        print('-------- Initializing App --------')
        screen_resolution = app.desktop().availableGeometry()
        self.title = 'Split Image Labeling tool'
        self.width, self.height = int(screen_resolution.width()), int(screen_resolution.height())
        self.setGeometry(0, 0, self.width, self.height)
        self.setWindowTitle(self.title)
        self.to_netcdf = netcdf

        # Load Tiff Image and split image data
        print('-------- Loading App Config --------')
        self.cfg_path = cfg_path
        with open(self.cfg_path, 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


        # Initialize dataset properties
        print('-------- Initializing Dataset --------')
        self.tiff_selector = 0
        self.checkpoint = checkpoint
        self.label_path = self.cfg['npy_path']
        self.label_data = np.load(self.label_path, allow_pickle=True)
        self.split_info_save = self.label_data[1]

        self.initDataset()


        print('-------- Initializing GUI --------')
        # Visualization toggles
        self.visualize_labels = False
        self.visualize_predictions = False
        self.visualize_heatmap = False

        # Initialize Image and Dimensions container variables
        bg_img_scaled = None
        self.bg_img_cv = None
        self.bg_qimage = None
        self.bg_img_utm = None
        self.image_index = None
        self.clicked = None
        self.conf_thresh = 0
        self.selected_classes = np.ones(len(self.class_enum))
        self.batch_select_polygon = []

        # Initialize colormaps (convert from hex if custom color map defined)
        if self.cfg['custom_color_map'] != 'None':
            colors = []
            for hex in self.cfg['custom_color_map']:
                colors.append(tuple(int(hex.lstrip('#')[i:i+2], 16)/256 for i in (0,2,4)))
            cmap = ListedColormap(colors)
            self.label_cmap = cmap
        else:
            self.label_cmap = get_cmap('tab20')
        self.conf_cmap = get_cmap('Spectral')
        colors = []
        for i in range(100):
            colors.append(tuple(self.conf_cmap(1 - i/100)[:-1]))
        self.conf_cmap = ListedColormap(colors)


        self.setMouseTracking(True)
        self.initUI()
        self.initBgImage()
        self.getNewImage(0)
        self.setLayout(self.master_layout)
        self.show()

    def initDataset(self):
        self.dataset_info = self.label_data[0]
        self.split_info = self.split_info_save[self.split_info_save[:,6] == self.tiff_selector]
        self.class_enum = self.dataset_info['class_enumeration']
        self.utm_epsg_code = self.cfg['utm_epsg_code']
        self.win_size = self.dataset_info['winsize_pix']
        self.lookup_tree = KDTree(self.split_info[:,2:4])

        # Load classification results if specified
        if self.checkpoint != None:
            self.pred_label_path = self.checkpoint
            #CST 20240313
            print("label path", self.pred_label_path)
            pred_data = np.load(self.pred_label_path, allow_pickle=True)
            self.pred_labels_save = pred_data[1]
            self.predictions = True
        else:
            self.predictions = False

        if self.checkpoint != None:
            self.pred_labels = self.pred_labels_save[self.pred_labels_save[:,6] == self.tiff_selector]

        self.geotiff = rio.open(self.dataset_info['filename'][self.tiff_selector])

        # Load Contour file if specified
        if self.cfg['contour_path'] != 'None':
            self.has_contour = True
            self.contour_np = np.load(self.cfg['contour_path'])
        else:
            self.has_contour = False
            self.contour_np = get_geotiff_bounds(self.geotiff, self.utm_epsg_code)
        self.contour_polygon = Polygon(self.contour_np)

        self.tiff_image_matrix = self.geotiff.read(1)

        self.tiff_image_max = get_img_sigma(self.tiff_image_matrix[::10,::10])

    
        

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
        self.conf_slider_container = QHBoxLayout()
        self.visualization_interactive = QHBoxLayout()
        self.visualization_toggles = QVBoxLayout()
        self.visualization_save_buttons = QVBoxLayout()
        self.split_text = QHBoxLayout()
        self.class_buttons = QVBoxLayout()

        # --- Initialize UI elements ---
        # Current split Image container
        self.split_image_pixmap = QPixmap()
        self.split_image_label = QLabel(self)
        self.split_image_label.setMargin(0)
        self.split_image_label.setAlignment(Qt.AlignCenter)

        # Background Image container
        self.tiff_image_pixmap = QPixmap()
        self.tiff_image_label = QLabel(self)
        self.tiff_image_label.setMargin(0)
        self.tiff_image_label.setMouseTracking(True)

        # Current class label container
        self.split_image_class = QLabel(self)
        self.split_image_class.setMargin(0)
        self.split_image_class.setAlignment(Qt.AlignCenter)
        #CST20240308
        font = QFont("Helvetica", 15)
        self.split_image_class.setFont(QFont("Helvetica", 15, QFont.Bold))

        # Current class confidence container
        self.split_image_conf = QLabel(self)
        self.split_image_conf.setMargin(0)
        self.split_image_conf.setAlignment(Qt.AlignCenter)
        #CST20240308
        self.split_image_conf.setFont(QFont("Helvetica", 15, QFont.Bold))

        # Buttons
        self.button_containers = []

        self.predictions_button = QCheckBox('Visualize Predictions', self)
        self.predictions_button.stateChanged.connect(self.predictionsCallback)

        self.heatmap_button = QCheckBox('Visualize Confidence Heatmap', self)
        self.heatmap_button.stateChanged.connect(self.heatmapCallback)

        self.labels_button = QCheckBox('Visualize Labels', self)
        self.labels_button.stateChanged.connect(self.labelsCallback)

        self.visualization_buttons = QButtonGroup(self)
        self.visualization_buttons.addButton(self.predictions_button,1)
        self.visualization_buttons.addButton(self.heatmap_button,2)
        self.visualization_buttons.addButton(self.labels_button,3)
        self.visualization_buttons.setExclusive(False)
        self.visualization_buttons.buttonClicked.connect(self.visualizationCallback)

        self.new_class_field = QLineEdit()
        self.new_class_field.textChanged.connect(self.textChangedCallback)
        self.new_class_field.returnPressed.connect(self.newClass)
        self.new_class_label = ''

        self.new_class_button = QPushButton('New Class')
        self.new_class_button.clicked.connect(self.newClassCallback)

        self.save_predictions_button = QPushButton('Save Predictions Image')
        self.save_predictions_button.clicked.connect(self.savePredictionsCallback)

        self.save_heatmap_button = QPushButton('Save Heatmap Image')
        self.save_heatmap_button.clicked.connect(self.saveHeatmapCallback)

        if self.has_contour == False:
            self.save_contour_button = QPushButton('Save Contour File')
            self.save_contour_button.clicked.connect(self.saveContourCallback)

        # Slider
        self.conf_thresh_value = QLabel(self)
        self.conf_thresh_value.setMargin(0)
        self.conf_thresh_value.setAlignment(Qt.AlignCenter)
        self.conf_thresh_value.setFont(QFont("Helvetica", 13, QFont.Bold))

        self.conf_thresh_slider = QSlider(Qt.Horizontal)
        self.conf_thresh_slider.setFocusPolicy(Qt.StrongFocus)
        self.conf_thresh_slider.setTickPosition(QSlider.TicksBothSides)
        self.conf_thresh_slider.setTickInterval(10)
        self.conf_thresh_slider.setSingleStep(1)
        self.conf_thresh_slider.setMinimum(0)
        self.conf_thresh_slider.setMaximum(99)
        self.conf_thresh_slider.setValue(0)
        low = np.floor(np.array(self.conf_cmap.colors[0])*255)
        mid = np.floor(np.array(self.conf_cmap.colors[50])*255)
        high = np.floor(np.array(self.conf_cmap.colors[99])*255)
        self.conf_thresh_slider.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba({},{},{},200), stop:0.5 rgba({},{},{},200), stop:1 rgba({},{},{},200)); \
                                               border-radius: 4px'.format(low[0],low[1],low[2],mid[0],mid[1],mid[2],high[0],high[1],high[2]))
        self.conf_thresh_slider.valueChanged.connect(self.slider_callback)

        self.initClassButtons()

        self.split_text.addSpacing(200)
        self.split_text.addWidget(self.split_image_class)
        self.split_text.addWidget(self.split_image_conf)
        self.split_text.addSpacing(200)

        self.visualization_widgets.addWidget(self.split_image_label)
        self.visualization_widgets.addLayout(self.split_text)

        self.conf_slider_container.addWidget(self.conf_thresh_slider)
        self.conf_slider_container.addWidget(self.conf_thresh_value)
        self.visualization_widgets.addLayout(self.conf_slider_container)

        self.visualization_toggles.addWidget(self.predictions_button)
        self.visualization_toggles.addWidget(self.heatmap_button)
        self.visualization_toggles.addWidget(self.labels_button)

        self.visualization_save_buttons.addWidget(self.save_predictions_button)
        self.visualization_save_buttons.addWidget(self.save_heatmap_button)
        if self.has_contour == False:
            self.visualization_save_buttons.addWidget(self.save_contour_button)

        self.visualization_interactive.addLayout(self.visualization_toggles)
        self.visualization_interactive.addLayout(self.visualization_save_buttons)

        self.visualization_widgets.addLayout(self.visualization_interactive)

        self.new_class_layout = QHBoxLayout()
        self.new_class_layout.addSpacing(200)
        self.new_class_layout.addWidget(self.new_class_field)
        self.new_class_layout.addSpacing(200)
        self.visualization_widgets.addLayout(self.new_class_layout)

        self.tiff_selector_buttons = QHBoxLayout()
        
        for tiffNum in range(len(self.dataset_info['filename'])):
            button = QPushButton('{}...'.format(self.dataset_info['filename'][tiffNum].split('/')[-1][:13]), self)
            button.clicked.connect(self.makeTiffSelectorCallbacks(tiffNum))
            button.clicked.connect(self.getTiffnum(tiffNum))
            self.tiff_selector_buttons.addWidget(button)

        self.left_layout.addLayout(self.tiff_selector_buttons)
        self.left_layout.addLayout(self.visualization_widgets)
        self.left_layout.addLayout(self.class_buttons)

        self.master_layout.addLayout(self.left_layout)
        self.master_layout.addWidget(self.tiff_image_label)

    def addClassButton(self, i, className, container):
        buttonContainer = QHBoxLayout()

        labelButton = QPushButton('{}: {}'.format(i, className), self)

        r,g,b,a = np.array(self.label_cmap(i))*255
        labelButton.setStyleSheet("background-color:rgb({},{},{});".format(r,g,b));
        #labelButton.setAlignment(Qt.AlignLeft)
        labelButton.clicked.connect(self.makeClassLabelCallbacks(i))

        labelToggle = QCheckBox(self)
        labelToggle.toggle()
        labelToggle.stateChanged.connect(self.makeClassToggleCallbacks(i))

        buttonContainer.addSpacing(20)
        buttonContainer.addWidget(labelToggle)
        buttonContainer.addWidget(labelButton)
        buttonContainer.addSpacing(20)
        container.addLayout(buttonContainer)

    def initClassButtons(self):

        numColumns = math.ceil(len(self.class_enum) / 12)

        self.class_buttons_columns_list = []
        for i in range(numColumns):
            self.class_buttons_columns_list.append(QVBoxLayout())

        for i, className in enumerate(self.class_enum):
            col = math.floor(i / 12)
            row = i % 12
            self.addClassButton(i, className, self.class_buttons_columns_list[col])

        self.class_buttons_columns = QHBoxLayout()
        for column in self.class_buttons_columns_list:
            self.class_buttons_columns.addLayout(column)

        self.class_buttons.addLayout(self.class_buttons_columns)

    def initBgImage(self):

        # Scale down tiff image for visualization and convert to 8-bit RGB
        scale_factor = int(self.tiff_image_matrix.shape[0] / 1200)
        bg_img_scaled = self.tiff_image_matrix[::scale_factor,::scale_factor]
        bg_img_scaled = scaleImage(bg_img_scaled, self.tiff_image_max)
        split_disp_size = (np.array(self.win_size) / scale_factor).astype('int') - 1
        bg_img_scaled = cv2.cvtColor(bg_img_scaled,cv2.COLOR_GRAY2RGB)

        # Draw split images on scaled down preview image
        if self.visualize_labels:
            draw = self.split_info[self.split_info[:,5] > self.conf_thresh]
            cmap = (np.array(self.label_cmap.colors)*255).astype(np.uint8)
            draw_split_image_labels(bg_img_scaled, scale_factor, split_disp_size, draw, self.selected_classes, cmap)

        elif self.visualize_predictions and self.predictions:
            draw = self.pred_labels[self.pred_labels[:,5] > self.conf_thresh]
            cmap = (np.array(self.label_cmap.colors)*255).astype(np.uint8)
            draw_split_image_labels(bg_img_scaled, scale_factor, split_disp_size, draw, self.selected_classes, cmap)

        elif self.visualize_heatmap and self.predictions:
            draw = self.pred_labels[self.pred_labels[:,5] > self.conf_thresh]
            cmap = (np.array(self.conf_cmap.colors)*255).astype(np.uint8)
            draw_split_image_confs(bg_img_scaled, scale_factor, split_disp_size, draw, self.selected_classes, cmap)

        # Rotate tiff to align North and plot glacier contour
        self.bg_img_cv, self.bg_img_utm, self.bg_img_transform = rotate_and_crop_geotiff(self.dataset_info, self.geotiff, bg_img_scaled, self.utm_epsg_code, self.contour_np, self.tiff_selector)
        #height,width,_ = bg_img_scaled.shape
        height,width,_ = self.bg_img_cv.shape

        # Convert to QImage from cv and wrap in QPixmap container
        #CST 20240312
        self.bg_qimg = QImage(self.bg_img_cv.data,self.bg_img_cv.shape[1],self.bg_img_cv.shape[0],self.bg_img_cv.shape[1]*3,QImage.Format_RGB888)
        self.tiff_image_pixmap = QPixmap(self.bg_qimg)
        self.tiff_image_pixmap = self.tiff_image_pixmap.scaledToWidth(int(self.width/2) - 10)
        #self.tiff_image_pixmap = self.tiff_image_pixmap.scaledToHeight(int(self.height - 50)).scaledToWidth(int(self.width/2) - 10)
        #print('bg_img_pixmap:  {}x{}'.format(self.tiff_image_pixmap.size().height(), self.tiff_image_pixmap.size().width()))
        # Get scaling factor between cv and q image
        bg_img_cv_size = np.array(self.bg_img_cv.shape[:-1])
        bg_img_q_size = np.array((self.tiff_image_pixmap.size().height(), self.tiff_image_pixmap.size().width()))
#CST 20240312
        self.scale_factor = bg_img_cv_size / bg_img_q_size
        self.tiff_image_label.setPixmap(QPixmap(self.tiff_image_pixmap))

    def updateBgImage(self):
        return

    def getNewImage(self, index):

         self.image_index = index

         # Grab info of split image at index
         x,y,x_utm,y_utm,label,conf,_ = self.split_info[index]
         x,y,x_utm,y_utm,label = int(x),int(y),int(x_utm),int(y_utm),int(label)
         #CST20240313
         # Get split image from image matrix
         img = self.tiff_image_matrix[x:x+self.win_size[0],y:y+self.win_size[1]]
         img = scaleImage(img, self.tiff_image_max)
         qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
         img = Image.fromarray(img).convert("L")
         img = ImageQt(img)
        

         # Wrap split image in QPixmap
         
         self.split_image_pixmap = QPixmap.fromImage(qimg).scaledToWidth(270)
         self.split_image_label.setPixmap(self.split_image_pixmap)
         # Update label text
         class_text = ''
         if self.predictions:
            label = int(self.pred_labels[index,4])
            conf = self.pred_labels[index,5]
            class_text = "Class %d: %s"%(label, self.class_enum[label])
            class_conf = "Conf: {:.2%}".format(conf)
         elif label == -1:
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
         cv2.circle(bg_img,(pix_coords[0][0],height-pix_coords[0][1]),4,(255,0,0),thickness=-1)
         cv2.line(bg_img, (pix_coords[0][0], 0), (pix_coords[0][0], height), (255,0,0),thickness=2)
         cv2.line(bg_img, (0, height-pix_coords[0][1]), (width, height-pix_coords[0][1]), (255,0,0),thickness=2)

         height,width,channels = bg_img.shape
         #CST20240312
         bg_img = QImage(bg_img.data,width,height,width*channels,QImage.Format_RGB888)
         background_image = QPixmap(bg_img)
         #background_image = background_image.scaledToHeight(int(self.height - 50)).scaledToWidth(int(self.width/2) - 10)
         background_image = background_image.scaledToWidth(int(self.width/2) - 10)

         self.tiff_image_label.setPixmap(background_image)

    #CST20240313 (creating a function to write images into a folder)
    def writeImage(self,filePath, fileName, index):
        p = os.path.join(filePath, fileName)
        fp = p + '.tif'
        self.image_index = index
         # Grab info of split image at index
        x,y,x_utm,y_utm,label,conf,_ = self.split_info[index]
        x,y,x_utm,y_utm,label = int(x),int(y),int(x_utm),int(y_utm),int(label)
         # Get split image from image matrix
        img = self.tiff_image_matrix[x:x+self.win_size[0],y:y+self.win_size[1]]
        img = scaleImage(img, self.tiff_image_max)
        image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        image.save(fp,"tif")
        

 #CST20240403 Checks all directories for the image, and deletes it if it finds the image.
    def deleteImage(self,filePath,fileName):
        numClasses = cfg['num_classes']
        for i in range(numClasses):
            file_path = (filePath+str(i)+'/'+str(i)+fileName+'.tif')
            if os.path.isfile(file_path):
                os.remove(file_path)
                
    
    def label(self, mask, class_label):
        self.split_info[mask,4] = class_label
        self.split_info[mask,5] = 1
        self.getNewImage(self.image_index)
        self.update()

    def labelCurrent(self, class_label):
        self.split_info[self.image_index][4] = class_label
        self.split_info[self.image_index][5] = 1
                #Load training img path
        if self.cfg['training_img_path'] != 'None':
            labeled_img_path = cfg['training_img_path']
            if not os.path.exists(labeled_img_path+"/"): os.mkdir(labeled_img_path+"/")
            if not os.path.exists(labeled_img_path+"/"+str(class_label)): os.mkdir(labeled_img_path+"/"+str(class_label))
            self.deleteImage(labeled_img_path+"/", str(self.image_index)+str(numTiff))
            self.writeImage(labeled_img_path+"/"+str(class_label), str(class_label)+str(self.image_index)+str(numTiff), self.image_index)
            self.getNewImage(self.image_index)
        else:
            if not os.path.exists("Classifications/"): os.mkdir("Classifications/")
            if not os.path.exists("Classifications/"+str(class_label)): os.mkdir("Classifications/"+str(class_label))
            self.deleteImage("Classifications/", str(self.image_index)+str(numTiff))
            self.writeImage("Classifications/"+str(class_label), str(class_label)+str(self.image_index)+str(numTiff), self.image_index)
            cfg['training_img_path'] = 'Classifications'
        self.update()

    def batchSelectLabel(self, class_label):
        height,width,_ = self.bg_img_cv.shape
        imgSize = np.array([width,height])
        pix_coords = utm_to_pix(imgSize, self.bg_img_utm.T, np.array(self.batch_select_polygon))
        batch_select = Polygon(self.batch_select_polygon)
        for i,img in enumerate(self.split_info):
            if Point(img[2],img[3]).within(batch_select):
                self.split_info[i][4] = class_label
                self.split_info[i][5] = 1
                if self.cfg['training_img_path'] != 'None':
                    labeled_img_path = self.cfg['training_img_path']
                    if not os.path.exists(labeled_img_path+"/"): os.mkdir(labeled_img_path+"/")
                    if not os.path.exists(labeled_img_path+"/"+str(class_label)): os.mkdir(labeled_img_path+"/"+str(class_label))
                    self.deleteImage(labeled_img_path+"/", str(i)+str(numTiff))
                    self.writeImage(labeled_img_path+"/"+str(class_label), str(class_label)+str(i)+str(numTiff), i)
                else:
                    if not os.path.exists("Classifications/"): os.mkdir("Classifications/")
                    if not os.path.exists("Classifications/"+str(class_label)): os.mkdir("Classifications/"+str(class_label))
                    self.deleteImage("Classifications/", str(i)+str(numTiff))
                    self.writeImage("Classifications/"+str(class_label), str(class_label)+str(i)+str(numTiff), i)
                    cfg['training_img_path'] = 'Classifications'
        self.batch_select_polygon = []
        self.getNewImage(self.image_index)
        self.update()


    def getMousePosUTM(self, event):

        # Get dimensions of the QT pixmap and window margins for the image preview
        width, height = self.tiff_image_label.size().width(), self.tiff_image_label.size().height()
        margin = height - self.tiff_image_pixmap.size().height()

        # Get the position of the mouse click in the GUI window
        click_pos = event.pos()

        # Map the click position in pixel space from GUI window to Image
        click_pos_scaled = self.tiff_image_label.mapFromParent(click_pos)

        # Correct for GUI window margins
        click_pos_corrected = np.array([click_pos_scaled.y() - int(margin/2), click_pos_scaled.x()])

        # Scale from image display size to underlying actual tiff image size
        click_pos_scaled = click_pos_corrected * self.scale_factor

        # Get UTM coordinates from tiff image pixel coordinates
        click_pos_utm = self.bg_img_transform * (click_pos_scaled[1], self.bg_img_cv.shape[0] - click_pos_scaled[0])

        return click_pos_utm

    def mousePressEvent(self, event):
        button = event.button()

        click_pos_utm = self.getMousePosUTM(event)

        # Left Click (move crosshairs)
        if button == 1:
            if Point(click_pos_utm[0], click_pos_utm[1]).within(self.contour_polygon):
                d,i = self.lookup_tree.query(click_pos_utm)
                self.getNewImage(i)

        # Right Click (draw polygon)
        elif button == 2:
            self.batch_select_polygon.append(click_pos_utm)

    def mouseMoveEvent(self, event):

        if self.batch_select_polygon != []:

            click_pos_utm = self.getMousePosUTM(event)

            bg_img = self.bg_img_cv.copy()
            height,width,_ = self.bg_img_cv.shape
            imgSize = np.array([width,height])
            pix_coords = utm_to_pix(imgSize, self.bg_img_utm.T, np.array(self.batch_select_polygon))
            current_pos = utm_to_pix(imgSize, self.bg_img_utm.T, np.array([[click_pos_utm[0], click_pos_utm[1]]]))

            for i in range(len(pix_coords)-1):
                cv2.circle(bg_img,(pix_coords[i][0],height-pix_coords[i][1]),4,(0,0,255),thickness=-1)
                cv2.line(bg_img, (pix_coords[i][0], height-pix_coords[i][1]), (pix_coords[i+1][0], height-pix_coords[i+1][1]), (0,0,255),thickness=2)

            cv2.line(bg_img, (pix_coords[-2][0], height-pix_coords[-2][1]), (current_pos[0][0], height-current_pos[0][1]), (255,0,0),thickness=2)

            height,width,channels = bg_img.shape
            #CST 202040312
            bg_img = QImage(bg_img.data,width,height,width*channels,QImage.Format_RGB888)
            background_image = QPixmap(bg_img)
            #background_image = background_image.scaledToHeight(int(self.height - 50)).scaledToWidth(int(self.width/2) - 10)
            background_image = background_image.scaledToWidth(int(self.width/2) - 10)
            #bg_img_scaled = background_image
            self.tiff_image_label.setPixmap(background_image)

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
            if key_map[event.key()] < len(self.class_enum):
                if self.batch_select_polygon != []:
                    self.batchSelectLabel(key_map[event.key()])
                else:
                    self.labelCurrent(key_map[event.key()])

        elif event.key() == 65: #Left arrow key
            if self.visualize_predictions or self.visualize_heatmap:
                index -= 1
                while self.pred_labels[index,5] < self.conf_thresh or not self.selected_classes[int(self.pred_labels[index,4])]:
                    index -= 1
            else:
                index -= 1
        elif event.key() == 68: #Right arrow key
            if self.visualize_predictions or self.visualize_heatmap:
                index += 1
                if index >= len(self.pred_labels):
                    index = 0
                while self.pred_labels[index,5] < self.conf_thresh or not self.selected_classes[int(self.pred_labels[index,4])]:
                    index += 1
                    if index >= len(self.pred_labels):
                        index = 0
            else:
                index += 1
            if index >= len(self.pred_labels):
                index = 0
        elif event.key() == Qt.Key_Escape:  #Escape key (deselect batch polygon)
            self.batch_select_polygon = []
        elif event.key() == 76: #l key - add current split image to training dataset
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier:
                #images_to_label = self.pred_labels[self.pred_labels[:,5] > self.conf_thresh]
                for i in range(len(self.selected_classes)):
                    if self.selected_classes[i]:
                        mask = (self.pred_labels[:,5] > self.conf_thresh) & (self.pred_labels[:,4] == i)
                        self.label(mask, i)

            else:
                _class = self.pred_labels[self.image_index][4]
                self.labelCurrent(_class)

        self.getNewImage(index)
        self.update()

    def makeClassLabelCallbacks(self, classLabel):
        def class_label_callback():
            self.labelCurrent(classLabel)
        return class_label_callback

    def makeClassToggleCallbacks(self, classLabel):
        def class_toggle_callback():
            self.selected_classes[classLabel] = not self.selected_classes[classLabel]
        return class_toggle_callback

    def makeTiffSelectorCallbacks(self, tiff_num):
        def tiff_selector_callback():
            self.split_info_save[self.split_info_save[:,6] == self.tiff_selector] = self.split_info
            self.tiff_selector = tiff_num
            self.initDataset()
            self.initBgImage()
            self.getNewImage(0)
        return tiff_selector_callback
    def getTiffnum(self, tiff):
        def getNum():
            global numTiff
            numTiff = tiff
        return getNum
    @pyqtSlot()
    def on_click(self):
        self.update()

    @pyqtSlot()
    def new_class(self):
        self.update()

    def visualizationCallback(self, id):
        self.initBgImage()

    def resetVisualization(self):
        self.visualize_labels = False
        self.visualize_predictions = False
        self.visualize_heatmap = False

    def predictionsCallback(self, state):
        if state == Qt.Checked:
            self.heatmap_button.setChecked(False)
            self.labels_button.setChecked(False)
            self.visualize_predictions = True
            self.visualize_heatmap = False
            self.visualize_labels = False
        else:
            self.resetVisualization()

    def heatmapCallback(self, state):
        if state == Qt.Checked:
            self.predictions_button.setChecked(False)
            self.labels_button.setChecked(False)
            self.visualize_predictions = False
            self.visualize_heatmap = True
            self.visualize_labels = False
        else:
            self.resetVisualization()

    def labelsCallback(self, state):
        if state == Qt.Checked:
            self.heatmap_button.setChecked(False)
            self.predictions_button.setChecked(False)
            self.visualize_predictions = False
            self.visualize_heatmap = False
            self.visualize_labels = True
        else:
            self.resetVisualization()


    def slider_callback(self):
        value = self.conf_thresh_slider.value()
        self.conf_thresh_value.setText('{}%'.format(value))
        self.conf_thresh = value/100.0

    def textChangedCallback(self, text):
        self.new_class_label = text

    def savePredictionsCallback(self):
        if self.predictions:
            self.predictionsCallback(Qt.Checked)
            self.initBgImage()
            out_path = self.pred_label_path[:-4] + '_prediction.png'
            cv2.imwrite(out_path, cv2.cvtColor(self.bg_img_cv, cv2.COLOR_BGR2RGB))
        #CST20240308
        else:
            out_path = self.pred_label_path[:-4] + '_prediction.png'
            print("savePredictionsCallback", out_path)


    def saveHeatmapCallback(self):
        if self.predictions:
            self.heatmapCallback(Qt.Checked)
            self.initBgImage()
            out_path = self.pred_label_path[:-4] + '_confidence_heatmap.png'
            cv2.imwrite(out_path, cv2.cvtColor(self.bg_img_cv, cv2.COLOR_BGR2RGB))
            #CST20240308
        else:
            out_path = self.pred_label_path[:-4] + '_confidence_heatmap.png'
            print("saveHeatmapCallback", out_path)

    def saveContourCallback(self):
        if self.batch_select_polygon != []:
            np.save(self.cfg_path[:-4]+'_contour.npy', np.array(self.batch_select_polygon))

    def newClass(self):
        self.class_enum.append(self.new_class_label)
        self.selected_classes = np.ones(len(self.class_enum))
        #self.addClassButton(len(self.class_enum)-1, self.new_class_label, self.class_buttons_columns_list[-1])
        self.clearLayout(self.class_buttons)
        self.initClassButtons()
        self.left_layout.addLayout(self.class_buttons)
        self.update()

    def newClassCallback(self):
        print('')

    def closeEvent(self, event):
        print('-------- Saving Data --------')
        self.split_info_save[self.split_info_save[:,6] == self.tiff_selector] = self.split_info
        self.label_data[1] = self.split_info_save
        #save_array = np.array([self.dataset_info, self.split_info], dtype=object)
        np.save(self.label_path, self.label_data)

        self.cfg['class_enum'] = self.class_enum
        self.cfg['num_classes'] = len(self.class_enum)
        f = open(args.config, 'w')
        f.write(generate_config_silas(self.cfg))
        f.close()

        if self.to_netcdf:
            save_array = np.array([self.dataset_info, self.pred_labels], dtype=object)
            to_netCDF(save_array, self.label_path[:-4])

        event.accept()
        return

if __name__ == '__main__':
    # Parse command line flags
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--load_labels", type=str, default=None)
    parser.add_argument("--netcdf", action="store_true")
    args = parser.parse_args()
    # Read config file
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    app = QApplication(sys.argv)
    QGuiApp = QApplication(sys.argv)
    ex = SplitImageTool(args.config, args.load_labels, args.netcdf)

    sys.exit(app.exec_())
