from utils import *
from auto_rotate_geotiff import *
#CST20240312
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QVBoxLayout, QCheckBox, QSlider, QLineEdit, QPushButton, QButtonGroup, QMainWindow, QGridLayout, QSizePolicy
#CST 20240308
from PyQt5.QtGui import QPixmap, QImage, QFont, QGuiApplication, QFont, qRgb

from PyQt5.QtCore import Qt

# from PIL.ImageQt import ImageQt
import os
import shutil


from Models import *
from Dataset import *

import rasterio as rio

from scipy.spatial import KDTree

from matplotlib.pyplot import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import yaml


class SplitImageTool(QWidget):
    global numTiff
    numTiff = 0
class SplitImageTool(QWidget):
    global numTiff
    numTiff = 0
    def __init__(self, cfg_path, checkpoint=None, netcdf=False):
        super().__init__()

        # Initialize GUI Window properties
        print('-------- Initializing App --------')
        
        # geometry
        
        # geometry
        screen_resolution = app.desktop().availableGeometry()
        self.title = 'Split Image Labeling tool'
        self.width, self.height = int(screen_resolution.width()), int(screen_resolution.height())
        self.setMinimumSize(self.width - 100, self.height - 100) # the default min size runs of the screen!
        self.setGeometry(0, 0, self.width - 100, self.height - 100)
        self.setMinimumSize(self.width - 100, self.height - 100) # the default min size runs of the screen!
        self.setGeometry(0, 0, self.width - 100, self.height - 100)
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
        self.lagstep = self.cfg['vario_num_lag']
        self.lagstep = self.cfg['vario_num_lag']
        self.label_data = np.load(self.label_path, allow_pickle=True)
        self.split_info_save = self.label_data[1]
        if self.split_info_save.ndim != 2 or self.split_info_save.shape[0] == 0:
            raise ValueError(
                "Dataset at {} has no split images; nothing to explore.".format(self.label_path)
            )
        # Only GeoTIFF indices that actually have splits in the .npy (others can be skipped by createDatasetFromGeotiff)
        self.tiff_indices_with_splits = np.unique(self.split_info_save[:, 6].astype(int))
        if self.tiff_selector not in self.tiff_indices_with_splits:
            self.tiff_selector = int(self.tiff_indices_with_splits[0])

        self.initDataset()


        print('-------- Initializing GUI --------')
        # Visualization toggles
        self.visualize_labels = False
        self.visualize_predictions = False
        self.visualize_heatmap = False

        # Initialize Image and Dimensions container variables
        self.downscale_big_image = False
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
            #CST 20240313
            print("label path", self.pred_label_path)
            pred_data = np.load(self.pred_label_path, allow_pickle=True)
            self.pred_labels_save = pred_data[1]
            self.predictions = True
        else:
            self.predictions = False

        if self.checkpoint != None:
            self.pred_labels = self.pred_labels_save[self.pred_labels_save[:,6] == self.tiff_selector]

        self.geotiff = rio.open(self.dataset_info['filename'][self.tiff_selector], nodata=65535)

        # Load Contour file if specified
        if self.cfg['contour_path'] != 'None':
            self.has_contour = True
            self.contour_np = np.load(self.cfg['contour_path'])
        else:
            self.has_contour = False
            self.contour_np = get_geotiff_bounds(self.geotiff, self.utm_epsg_code)
        self.contour_polygon = Polygon(self.contour_np)

        raw_band = self.geotiff.read(1)
        nodata_set = wv_geotiff_nodata_set(self.geotiff.nodata)
        valid_samples = valid_raster_samples(raw_band, stride=10, nodata_set=nodata_set)
        self.tiff_image_max = get_img_sigma(valid_samples) if valid_samples.size >= 1 else 1.0
        self.tiff_image_matrix = mask_nodata_for_display(raw_band, nodata_set)

        # Percentile-based contrast bounds for the big-map preview only (patch preview
        # keeps scaleImage so labeled thumbnails stay consistent across sessions).
        if valid_samples.size >= 2:
            lo, hi = np.percentile(valid_samples, [1.0, 99.0])
        else:
            lo, hi = 0.0, float(self.tiff_image_max)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, max(1.0, float(self.tiff_image_max))
        self.tiff_display_lo = float(lo)
        self.tiff_display_hi = float(hi)

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

    def right_column_image_width(self):
        """Pixel width for the split preview and geotiff in the right column (shared for layout + mouse mapping)."""
        return max(420, min(1600, int(self.width * 0.46) - 24))

    def variogram_thumb_width(self):
        w = self.right_column_image_width()
        return max(220, min(420, w // 2 - 8))

    def split_preview_fixed_height(self):
        """Fixed QLabel height for the patch preview (screen-based); pixmap scales inside without vertical clipping."""
        return max(260, min(520, int(self.height * 0.28)))

    def initUI(self):

        # Ensure the main widget always receives key events (WASD / arrow navigation,
        # number keys, L) even after the user clicks a button or slider.
        self.setFocusPolicy(Qt.StrongFocus)

        self.master_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.visualization_widgets = QVBoxLayout()
        self.conf_slider_container = QHBoxLayout()
        self.visualization_interactive = QHBoxLayout()
        self.visualization_toggles = QVBoxLayout()
        self.visualization_save_buttons = QVBoxLayout()
        self.split_text = QHBoxLayout()
        self.class_buttons = QVBoxLayout()
        self.grid_layout = QGridLayout()

        # --- Initialize UI elements ---
        # Current split Image container
        self.split_image_pixmap = QPixmap()
        self.split_image_label = QLabel(self)
        self.split_image_label.setMargin(0)
        self.split_image_label.setAlignment(Qt.AlignCenter)
        self.split_image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.split_image_label.setFixedHeight(self.split_preview_fixed_height())
        #self.split_image_labels = [QLabel() for _ in self.lagstep]
        self.split_image_labels = [QLabel(self) for _ in range(4)]
        for i in range(4):
            self.split_image_labels[i].setMargin(0)
            self.split_image_labels[i].setAlignment(Qt.AlignCenter)

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

        # UTM at crosshair (same anchor as createDatasetFromGeotiff: upper-left of split window in UTM)
        self.crosshair_utm_label = QLabel(self)
        self.crosshair_utm_label.setMargin(0)
        self.crosshair_utm_label.setAlignment(Qt.AlignCenter)
        self.crosshair_utm_label.setFont(QFont("Helvetica", 11))
        self.crosshair_utm_label.setWordWrap(True)

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

         #SAVE CONFIDENCE PREDICTIONS
        self.save_confidente_predictions_button = QPushButton('Save Confidence Predictions')
        self.save_confidente_predictions_button.clicked.connect(self.savePredictionsCallbackNPY)

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

        self.split_text.addStretch(1)
        self.split_text.addWidget(self.split_image_class)
        self.split_text.addWidget(self.split_image_conf)
        self.split_text.addStretch(1)

        self.conf_slider_container.addWidget(self.conf_thresh_slider)
        self.conf_slider_container.addWidget(self.conf_thresh_value)
        self.visualization_widgets.addLayout(self.conf_slider_container)
        

        self.visualization_toggles.addWidget(self.predictions_button)
        self.visualization_toggles.addWidget(self.heatmap_button)
        self.visualization_toggles.addWidget(self.labels_button)

        self.visualization_save_buttons.addWidget(self.save_predictions_button)
        #INITIALIZE THE BUTTON THAT SAVES THE CONFIDENT PREDICTIONS 
        self.visualization_save_buttons.addWidget(self.save_confidente_predictions_button)
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
        

        self.tiff_selector_buttons = QGridLayout()

        for grid_idx, tiffNum in enumerate(self.tiff_indices_with_splits):
            button = QPushButton('{}...'.format(self.dataset_info['filename'][tiffNum].split('/')[-1][:13]), self)
            button.clicked.connect(self.makeTiffSelectorCallbacks(int(tiffNum)))
            button.clicked.connect(self.getTiffnum(int(tiffNum)))
            #self.tiff_selector_buttons.addWidget(button)
            
            # Calculate the row and column numbers
            row = grid_idx // 4
            col = grid_idx % 4
            
            # Add the button to the layout at the specified row and column
            self.tiff_selector_buttons.addWidget(button, row, col)

        self.left_layout.addLayout(self.tiff_selector_buttons)
        
        self.left_layout.addLayout(self.visualization_widgets)
        
        self.left_layout.addLayout(self.class_buttons)
        
        self.left_layout.addLayout(self.grid_layout) #something about this causes a warning: QLayout::addChildLayout: layout "" already has a parent
        # this might be the issue? as in the layout is added to the master too early?

        # Right column: class/conf, fixed-height split preview, then geotiff (extra height goes to the map)
        self.right_layout.addLayout(self.split_text)
        self.right_layout.addWidget(self.crosshair_utm_label)
        self.right_layout.addWidget(self.split_image_label, stretch=0)
        self.right_layout.addWidget(self.tiff_image_label, stretch=1)

        self.master_layout.addLayout(self.left_layout, stretch=0)
        self.master_layout.addLayout(self.right_layout, stretch=1)

        


        


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

        numColumns = math.ceil(len(self.class_enum) / 4)

        self.class_buttons_columns_list = []
        for i in range(numColumns):
            self.class_buttons_columns_list.append(QVBoxLayout())

        for i, className in enumerate(self.class_enum):
            col = math.floor(i / 4)
            row = i % 4
            self.addClassButton(i, className, self.class_buttons_columns_list[col])

        self.class_buttons_columns = QHBoxLayout()
        for column in self.class_buttons_columns_list:
            self.class_buttons_columns.addLayout(column)

        self.class_buttons.addLayout(self.class_buttons_columns)
    
    def _big_image_target_size(self):
        """Target (width, height) in physical pixels for the big-map render.

        Scales to the on-screen pixmap width (logical points) times the device pixel
        ratio so HiDPI displays get a crisp image, while preserving the source aspect
        ratio. Never upscales above the source resolution.
        """
        src_h, src_w = self.tiff_image_matrix.shape[:2]
        tw_logical = int(self.right_column_image_width())
        if self.downscale_big_image:
            tw_logical = max(240, tw_logical // 2)
        dpr = float(self.devicePixelRatioF()) if hasattr(self, 'devicePixelRatioF') else 1.0
        dpr = max(1.0, dpr)
        target_w = max(1, int(round(tw_logical * dpr)))
        if src_w > 0:
            target_h = max(1, int(round(target_w * (src_h / float(src_w)))))
        else:
            target_h = 1
        if target_w > src_w or target_h > src_h:
            target_w, target_h = src_w, src_h
        return target_w, target_h, dpr

    def _stretch_for_display(self, arr):
        """Apply a cached percentile stretch and return uint8 grayscale for display."""
        lo = getattr(self, 'tiff_display_lo', 0.0)
        hi = getattr(self, 'tiff_display_hi', float(self.tiff_image_max))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return scaleImage(arr, max(1.0, float(self.tiff_image_max)))
        a = np.asarray(arr, dtype=np.float32)
        a = (a - float(lo)) / (float(hi) - float(lo))
        np.clip(a, 0.0, 1.0, out=a)
        return (a * 255.0 + 0.5).astype(np.uint8)

    def initBgImage(self):
        target_w, target_h, dpr = self._big_image_target_size()
        src_h, src_w = self.tiff_image_matrix.shape[:2]

        # Anti-aliased downsample straight to the on-screen physical pixel size.
        bg_img_scaled = cv2.resize(
            self.tiff_image_matrix,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
        bg_img_scaled = self._stretch_for_display(bg_img_scaled)
        bg_img_scaled = cv2.cvtColor(bg_img_scaled, cv2.COLOR_GRAY2RGB)

        # Ratio between raster pixels and the scaled preview (float). Used by the
        # label/heatmap overlay helpers to convert split window size + centre coords.
        raster_to_scaled = src_h / float(target_h) if target_h > 0 else 1.0
        split_disp_size = np.maximum(
            1, (np.array(self.win_size) / raster_to_scaled).astype('int') - 1
        )

        # Draw split images on scaled down preview image
        if self.visualize_labels:
            draw = self.split_info[self.split_info[:,5] > self.conf_thresh]
            cmap = (np.array(self.label_cmap.colors)*255).astype(np.uint8)
            draw_split_image_labels(bg_img_scaled, raster_to_scaled, split_disp_size, draw, self.selected_classes, cmap)

        elif self.visualize_predictions and self.predictions:
            draw = self.pred_labels[self.pred_labels[:,5] > self.conf_thresh]
            cmap = (np.array(self.label_cmap.colors)*255).astype(np.uint8)
            draw_split_image_labels(bg_img_scaled, raster_to_scaled, split_disp_size, draw, self.selected_classes, cmap)

        elif self.visualize_heatmap and self.predictions:
            draw = self.pred_labels[self.pred_labels[:,5] > self.conf_thresh]
            cmap = (np.array(self.conf_cmap.colors)*255).astype(np.uint8)
            draw_split_image_confs(bg_img_scaled, raster_to_scaled, split_disp_size, draw, self.selected_classes, cmap)

        # Rotate tiff to align North and plot glacier contour
        self.bg_img_cv, self.bg_img_utm, self.bg_img_transform = rotate_and_crop_geotiff(self.dataset_info, self.geotiff, bg_img_scaled, self.utm_epsg_code, self.contour_np, self.tiff_selector)

        # Convert to QImage from cv and wrap in QPixmap container. bg_img_cv is already
        # at display resolution, so set the device pixel ratio and let Qt render 1:1.
        self.bg_qimg = QImage(
            self.bg_img_cv.data,
            self.bg_img_cv.shape[1],
            self.bg_img_cv.shape[0],
            self.bg_img_cv.shape[1] * 3,
            QImage.Format_RGB888,
        )
        self.tiff_image_pixmap = QPixmap.fromImage(self.bg_qimg)

        tw_logical = int(self.right_column_image_width())
        if self.downscale_big_image:
            tw_logical = max(240, tw_logical // 2)
        desired_physical_w = max(1, int(round(tw_logical * dpr)))
        if self.tiff_image_pixmap.width() != desired_physical_w:
            self.tiff_image_pixmap = self.tiff_image_pixmap.scaledToWidth(
                desired_physical_w, Qt.SmoothTransformation
            )
        self.tiff_image_pixmap.setDevicePixelRatio(dpr)

        # Map from bg_img_cv (physical) coords to the logical pixmap coords used by
        # mouse events; getMousePosUTM multiplies click_pos (logical) by scale_factor.
        bg_img_cv_size = np.array(self.bg_img_cv.shape[:-1], dtype=np.float64)
        pm_physical = np.array(
            (self.tiff_image_pixmap.size().height(), self.tiff_image_pixmap.size().width()),
            dtype=np.float64,
        )
        pm_logical = pm_physical / dpr
        self.scale_factor = bg_img_cv_size / np.maximum(pm_logical, 1.0)
        self.tiff_image_label.setPixmap(self.tiff_image_pixmap)

    def updateBgImage(self):
        return
    #For graphing variograms
    def draw_line(self, img, x1, y1, x2, y2, color):
        """Draw a simple line on a QImage."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < img.width() and 0 <= y1 < img.height():
                img.setPixel(x1, y1, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def draw_variogram_on_image(self, vario, qimage):
        width = qimage.width()
        height = qimage.height()
        
        # Normalize vario values to fit within image dimensions
        max_value = max(vario)
        min_value = min(vario)
        
        for i in range(len(vario) - 1):
            x1 = int(i * width / len(vario))
            y1 = int((vario[i] - min_value) * height / (max_value - min_value))
            x2 = int((i + 1) * width / len(vario))
            y2 = int((vario[i + 1] - min_value) * height / (max_value - min_value))
            
            # Ensure coordinates are within bounds
            y1 = height - min(max(y1, 0), height - 1)
            y2 = height - min(max(y2, 0), height - 1)
            
            # Draw line on the image (basic line drawing algorithm)
            self.draw_line(qimage, x1, y1, x2, y2, qRgb(0, 0, 0))

    def _directional_vario_lag_steps(self, split_shape):
        """Return (lagStepNS, lagStepEW, lagStepDiag) matching silas_directional_vario."""
        if split_shape[0] < split_shape[1]:
            return 3, 4, 5
        elif split_shape[0] > split_shape[1]:
            return 4, 3, 5
        else:
            # Square patch: silas_directional_vario leaves lag steps undefined in this
            # branch; pick 1 so lag axes are still in pixel units and monotonic.
            return 1, 1, 1

    def _variogram_label_size(self):
        """(width, height) for each variogram thumbnail in the 2x2 grid."""
        w = self.variogram_thumb_width()
        return w, max(160, int(w * 0.75))

    def _render_variogram_pixmap(self, vario, lag_step, title):
        """Render a directional variogram as a matplotlib figure and return a QPixmap."""
        width_px, height_px = self._variogram_label_size()
        v = np.asarray(vario, dtype=np.float64).ravel()
        # silas_directional_vario right-pads each row with zeros so all four directions
        # share the same column count; trim trailing exact zeros so we don't plot those.
        if v.size > 0:
            nonzero = np.flatnonzero(v != 0.0)
            last = int(nonzero[-1]) + 1 if nonzero.size else 0
            v = v[:last]

        dpi = 100.0
        fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, tight_layout=True)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        if v.size > 0 and np.any(np.isfinite(v)):
            lags = np.arange(1, v.size + 1, dtype=np.float64) * float(lag_step)
            ax.plot(lags, v, color='tab:blue', linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Lag h (pixels)", fontsize=8)
        ax.set_ylabel("Semi-variance", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.3)

        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4).copy()
        qimg = QImage(buf.data, w, h, w * 4, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    def _overlay_scale(self):
        # self.scale_factor is [cv_h / q_h, cv_w / q_w]. When downscale_big_image is False
        # this can be ~10-17x, so overlays drawn with fixed pixel thickness on bg_img_cv
        # collapse to sub-pixel sizes once Qt shrinks the pixmap for display. Multiplying
        # drawing constants by this factor keeps crosshairs/outlines visually consistent.
        sf = getattr(self, 'scale_factor', None)
        if sf is None:
            return 1.0
        return max(1.0, float(np.mean(sf)))

    def _get_selected_split_utm_corners(self):
        if self.image_index is None or len(self.split_info) == 0:
            return None

        x, y = self.split_info[self.image_index][:2]
        x = int(x)
        y = int(y)
        split_h = int(self.win_size[0])
        split_w = int(self.win_size[1])

        raster_corners = np.array(
            [
                [x, y],
                [x + split_h - 1, y],
                [x + split_h - 1, y + split_w - 1],
                [x, y + split_w - 1],
            ],
            dtype=np.float64,
        )
        utm_corners = np.array(
            [self.geotiff.xy(int(r), int(c)) for r, c in raster_corners], dtype=np.float64
        )
        return utm_corners

    def _draw_selected_split_boundary(self, bg_img, color=(0, 255, 255), thickness=None):
        split_utm_corners = self._get_selected_split_utm_corners()
        if split_utm_corners is None:
            return

        if thickness is None:
            thickness = max(1, int(round(2 * self._overlay_scale())))

        height, width, _ = bg_img.shape
        img_size = np.array([width, height])
        pix_corners = utm_to_pix(img_size, self.bg_img_utm.T, split_utm_corners)
        for i in range(len(pix_corners)):
            p0 = pix_corners[i]
            p1 = pix_corners[(i + 1) % len(pix_corners)]
            cv2.line(
                bg_img,
                (int(p0[0]), int(height - p0[1])),
                (int(p1[0]), int(height - p1[1])),
                color,
                thickness=thickness,
            )

    def _set_big_image_pixmap(self, bg_img):
        height, width, channels = bg_img.shape
        qimg = QImage(bg_img.data, width, height, width * channels, QImage.Format_RGB888)
        background_image = QPixmap.fromImage(qimg)
        dpr = float(self.devicePixelRatioF()) if hasattr(self, 'devicePixelRatioF') else 1.0
        dpr = max(1.0, dpr)
        tw_logical = int(self.right_column_image_width())
        if self.downscale_big_image:
            tw_logical = max(240, tw_logical // 2)
        desired_physical_w = max(1, int(round(tw_logical * dpr)))
        if background_image.width() != desired_physical_w:
            background_image = background_image.scaledToWidth(
                desired_physical_w, Qt.SmoothTransformation
            )
        background_image.setDevicePixelRatio(dpr)
        self.tiff_image_label.setPixmap(background_image)

    def getNewImage(self, index):
        
        self.image_index = index

         # Grab info of split image at index
        x, y, x_utm, y_utm, label, conf, _ = self.split_info[index]
        x, y, label = int(x), int(y), int(label)
        e_utm = float(x_utm)
        n_utm = float(y_utm)
        self.crosshair_utm_label.setText(
            "Crosshair UTM (EPSG {}):  E = {:.2f} m,  N = {:.2f} m".format(
                int(self.utm_epsg_code), e_utm, n_utm
            )
        )
        #CST20240313
        # Split from masked full raster (no 65535). Float avoids uint wraparound in lag differences.
        raw_split = self.tiff_image_matrix[x:x+self.win_size[0],y:y+self.win_size[1]]
        img = scaleImage(raw_split, self.tiff_image_max)
        variograms = silas_directional_vario(np.asarray(raw_split, dtype=np.float64))
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        image = Image.fromarray(img).convert("L")
         # Wrap split image in QPixmap       
        mw = self.right_column_image_width()
        mh = self.split_preview_fixed_height()
        self.split_image_pixmap = QPixmap.fromImage(qimg).scaled(
            mw, mh, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.split_image_label.setPixmap(self.split_image_pixmap)
        

        self.variograms = variograms

        lagStepNS, lagStepEW, lagStepDiag = self._directional_vario_lag_steps(raw_split.shape)
        vario_meta = [
            ("North-South variogram", lagStepNS),
            ("East-West variogram", lagStepEW),
            ("NE-SW diagonal variogram", lagStepDiag),
            ("NW-SE diagonal variogram", lagStepDiag),
        ]
        thumb_w, thumb_h = self._variogram_label_size()
        for i, vario in enumerate(self.variograms):
            title, lag_step = vario_meta[i]
            pm = self._render_variogram_pixmap(vario, lag_step, title)

            row = i // 2
            col = i % 2

            vario_label = self.split_image_labels[i]
            vario_label.setFixedSize(thumb_w, thumb_h)
            vario_label.setPixmap(pm)
            self.grid_layout.addWidget(vario_label, row, col)
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

        pix_coords = utm_to_pix(imgSize, self.bg_img_utm.T, np.array([[e_utm, n_utm]]))

        # Draw crosshairs (scaled so overlays keep a consistent on-screen size regardless
        # of downscale_big_image toggle)
        s = self._overlay_scale()
        dot_radius = max(2, int(round(4 * s)))
        line_thickness = max(1, int(round(2 * s)))
        cv2.circle(bg_img,(pix_coords[0][0],height-pix_coords[0][1]),dot_radius,(255,0,0),thickness=-1)
        cv2.line(bg_img, (pix_coords[0][0], 0), (pix_coords[0][0], height), (255,0,0),thickness=line_thickness)
        cv2.line(bg_img, (0, height-pix_coords[0][1]), (width, height-pix_coords[0][1]), (255,0,0),thickness=line_thickness)
        self._draw_selected_split_boundary(bg_img)

        self._set_big_image_pixmap(bg_img)

    #CST20240313 (creating a function to write images into a folder)
    def writeImage(self,filePath, fileName, index):
        p = os.path.join(filePath, fileName)
        fp = p + '.png'
        self.image_index = index
         # Grab info of split image at index
        x,y,x_utm,y_utm,label,conf,_ = self.split_info[index]
        x,y,x_utm,y_utm,label = int(x),int(y),int(x_utm),int(y_utm),int(label)
         # Get split image from image matrix
        img = self.tiff_image_matrix[x:x+self.win_size[0],y:y+self.win_size[1]]
        img = scaleImage(img, self.tiff_image_max)
        image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        image.save(fp,"png")
        

 #CST20240403 Checks all directories for the image, and deletes it if it finds the image.
    def deleteImage(self,filePath,fileName):
        numClasses = cfg['num_classes']
        for i in range(numClasses):
            file_path = (filePath+str(i)+'/'+str(i)+fileName+'.png')
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
            self._draw_selected_split_boundary(bg_img)

            s = self._overlay_scale()
            dot_radius = max(2, int(round(4 * s)))
            line_thickness = max(1, int(round(2 * s)))
            for i in range(len(pix_coords)-1):
                cv2.circle(bg_img,(pix_coords[i][0],height-pix_coords[i][1]),dot_radius,(0,0,255),thickness=-1)
                cv2.line(bg_img, (pix_coords[i][0], height-pix_coords[i][1]), (pix_coords[i+1][0], height-pix_coords[i+1][1]), (0,0,255),thickness=line_thickness)

            cv2.line(bg_img, (pix_coords[-2][0], height-pix_coords[-2][1]), (current_pos[0][0], height-current_pos[0][1]), (255,0,0),thickness=line_thickness)
            self._set_big_image_pixmap(bg_img)

    def _neighbor_in_direction(self, direction):
        """Return the index of the spatially nearest split in the given direction.

        direction is one of 'up', 'down', 'left', 'right' (screen-aligned, which after
        the north-up rotation maps to UTM N/S/E/W). If no candidate exists in that
        direction, returns the current index (no wrap-around at edges).
        """
        if self.image_index is None or len(self.split_info) == 0:
            return self.image_index

        E0 = float(self.split_info[self.image_index, 2])
        N0 = float(self.split_info[self.image_index, 3])
        dE = self.split_info[:, 2].astype(np.float64) - E0
        dN = self.split_info[:, 3].astype(np.float64) - N0

        if direction == 'up':
            cone = (dN > 0) & (np.abs(dE) < dN)
            parallel = dN
            perpendicular = np.abs(dE)
        elif direction == 'down':
            cone = (dN < 0) & (np.abs(dE) < -dN)
            parallel = -dN
            perpendicular = np.abs(dE)
        elif direction == 'right':
            cone = (dE > 0) & (np.abs(dN) < dE)
            parallel = dE
            perpendicular = np.abs(dN)
        elif direction == 'left':
            cone = (dE < 0) & (np.abs(dN) < -dE)
            parallel = -dE
            perpendicular = np.abs(dN)
        else:
            return self.image_index

        if self.visualize_predictions or self.visualize_heatmap:
            if self.predictions and len(self.pred_labels) == len(self.split_info):
                conf_ok = self.pred_labels[:, 5] >= self.conf_thresh
                class_ok = np.array(
                    [bool(self.selected_classes[int(c)]) for c in self.pred_labels[:, 4]],
                    dtype=bool,
                )
                cone &= conf_ok & class_ok

        candidate_idx = np.flatnonzero(cone)
        if candidate_idx.size == 0:
            return self.image_index

        # Prefer the candidate closest to the cardinal axis (smallest perpendicular
        # offset); break ties by the smallest step along the direction. This selects
        # the immediate neighbour on a regular grid even when the UTM rows/columns
        # are slightly irregular.
        perp = perpendicular[candidate_idx]
        par = parallel[candidate_idx]
        order = np.lexsort((par, perp))
        return int(candidate_idx[order[0]])

    def keyPressEvent(self, event):
        key = event.key()

        if 48 <= key <= 57:
            label_idx = key - 48
            if label_idx < len(self.class_enum):
                if self.batch_select_polygon != []:
                    self.batchSelectLabel(label_idx)
                else:
                    self.labelCurrent(label_idx)
            self.update()
            return

        if key in (Qt.Key_Left, Qt.Key_A):
            new_index = self._neighbor_in_direction('left')
        elif key in (Qt.Key_Right, Qt.Key_D):
            new_index = self._neighbor_in_direction('right')
        elif key in (Qt.Key_Up, Qt.Key_W):
            new_index = self._neighbor_in_direction('up')
        elif key in (Qt.Key_Down, Qt.Key_S):
            new_index = self._neighbor_in_direction('down')
        elif key == Qt.Key_Escape:
            self.batch_select_polygon = []
            self.update()
            return
        elif key == Qt.Key_L:
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier:
                for i in range(len(self.selected_classes)):
                    if self.selected_classes[i]:
                        mask = (self.pred_labels[:, 5] > self.conf_thresh) & (self.pred_labels[:, 4] == i)
                        self.label(mask, i)
            else:
                _class = self.pred_labels[self.image_index][4]
                self.labelCurrent(_class)
            self.update()
            return
        else:
            return

        if new_index is not None and new_index != self.image_index:
            self.getNewImage(new_index)
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
            print("GEOTIFF number", numTiff)
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
            ("Prediction image saved to", out_path)


    def saveHeatmapCallback(self):
        if self.predictions:
            self.heatmapCallback(Qt.Checked)
            self.initBgImage()
            out_path = self.pred_label_path[:-4] + '_confidence_heatmap.png'
            cv2.imwrite(out_path, cv2.cvtColor(self.bg_img_cv, cv2.COLOR_BGR2RGB))
            #CST20240308
        else:
            out_path = self.pred_label_path[:-4] + '_confidence_heatmap.png'
            print("Heat map saved to", out_path)

    #function that saves the confident predictions
    def savePredictionsCallbackNPY(self):
        savepred = cfg['save_all_pred']
        saveMin = cfg['equal_dataset']
    
        #check if there are predictions loaded so that app doesn't crash
        if self.checkpoint == None:
            print('No predictions loaded')
            return
        #create the directory if it does not exist
        dirName = 'ConfidentPredictions'
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        #create the filename
        filename = f'ConfidentPredictions/confidence_predictions_{self.conf_thresh}.npy'

        #get the dataset
        dataset_path = cfg['npy_path']   
        dataset = np.load(dataset_path, allow_pickle=True)
        if saveMin == False:
            if savepred == False:
                # Save predicitions above the confidence threshold
                self.confident_predictions = self.pred_labels[self.pred_labels[:,5] > self.conf_thresh]        

                dataset[1] = self.confident_predictions #update the dataset with the new confident predictions
                
            else: #Should save all WV datasets, not just the one selected.
                self.confident_predictions = self.pred_labels_save[self.pred_labels_save[:,5] > self.conf_thresh]
                dataset[1] = self.confident_predictions
            np.save(filename, dataset) #save the dataset as npy file
            print('File saved to', filename)
        else:
            total = 0
            numClasses = cfg['num_classes']
            minSize =  100000
            classSize = 0
            classes = 0
            predictions = []
            if savepred == False:
                
                # Save predicitions above the confidence threshold
                for i in range(numClasses):
                    data = self.pred_labels[self.pred_labels[:,4] == i]
                    self.confident_predictions = data[data[:,5] > self.conf_thresh]
                    classSize = len(self.confident_predictions)
                    if classSize < minSize: 
                        minSize = classSize
                for i in range(numClasses): 
                    data = self.pred_labels[self.pred_labels[:,4] == i]
                    self.confident_predictions = data[data[:,5] > self.conf_thresh]
                    if classSize != 0:
                        for i in range(minSize): #Should select the highest confidence images from each class
                                highest_confidence_index = np.argmax(self.confident_predictions[:, 5])
                                predictions.append(self.confident_predictions[highest_confidence_index])
                                self.confident_predictions = np.delete(self.confident_predictions, highest_confidence_index, axis=0)
                                total += 1
                predictions = np.array(predictions)
                dataset[1] = predictions #update the dataset with the new confident predictions
                
            else: #Should save all WV datasets, not just the one selected.
                # Save predicitions above the confidence threshold
                for i in range(numClasses):
                    data = self.pred_labels_save[self.pred_labels_save[:,4] == i]
                    self.confident_predictions = data[data[:,5] > self.conf_thresh]
                    classSize = len(self.confident_predictions)
                    if classSize != 0:
                        print("Saving images from class ", i)
                        print("Class: ", i, "Number of Images: ", classSize)
                        classes += 1
                        if classSize < minSize: 
                            minSize = classSize                           
                for i in range(numClasses): 
                    data = self.pred_labels_save[self.pred_labels_save[:,4] == i]
                    self.confident_predictions = data[data[:,5] > self.conf_thresh]
                    classSize = len(self.confident_predictions)
                    if classSize != 0:
                        for i in range(minSize):  #Should select the highest confidence images from each class
                            highest_confidence_index = np.argmax(self.confident_predictions[:, 5])
                            predictions.append(self.confident_predictions[highest_confidence_index])
                            self.confident_predictions = np.delete(self.confident_predictions, highest_confidence_index, axis=0)
                            total += 1
                predictions = np.array(predictions)
                dataset[1] = predictions #update the dataset with the new confident predictions
            print(minSize, "Images saved in for each class for a total dataset size of ", total, "images from ", classes, "crevasse classes")
            np.save(filename, dataset) #save the dataset as npy file
            print('File saved to', filename)


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
