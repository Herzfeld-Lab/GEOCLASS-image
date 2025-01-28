import sys
sys.path.append('/home/twickler/ws/GEOCLASS-image/NN_Class')
import numpy as np
from matplotlib.pyplot import get_cmap
from PIL import Image
import matplotlib.pyplot as plt

label_cmap = get_cmap('tab20')
plot = Image.open('/home/twickler/plots/pass0_Fig1_nrb_data_valid_mask.png') #for multiple images have [self.tiff_selector] inside ()
tiff_image_matrix = np.array(plot)
selected_classes = np.ones(4)
split_disp_size = [25,25]
pred_label_path = 'Output/calipso_config.config_19-11-2024_17:28/labels/labeled_epoch_5.npy'
pred_data = np.load(pred_label_path, allow_pickle=True)
pred_labels = pred_data[1]


img_mat = tiff_image_matrix
img_mat = img_mat[:, :, :3]
labels = pred_labels
labels = np.delete(labels, 4, axis=1)
cmap = (np.array(label_cmap.colors)*255).astype(np.uint8)

for i, selected_class in enumerate(selected_classes):
        if selected_class:
            # Select the rows corresponding to the class
            clas = labels[labels[:, 2] == i]  # Assuming labels[:, 2] holds class information
            c = cmap[i]
            x = np.floor(clas[:,0]).reshape(-1,1).astype(np.int32)
            y = np.floor(clas[:,1]).reshape(-1,1).astype(np.int32)
            #y = int(img_mat.shape[1]) - y1
            xy = np.concatenate((x,y),axis=1)
            for splitImg in xy:
                x_start, y_start = splitImg[0], splitImg[1]
                x_end = min(x_start + split_disp_size[0], img_mat.shape[1])
                y_end = min(img_mat.shape[0] - y_start, img_mat.shape[0])  # Flip the y-coordinate

                # Ensure coordinates are within bounds
                if 0 <= x_start < img_mat.shape[1] and 0 <= y_start < img_mat.shape[0]:
                    img_mat[y_end-split_disp_size[1]:y_end, x_start:x_end, 0] = c[0]
                    img_mat[y_end-split_disp_size[1]:y_end, x_start:x_end, 1] = c[1]
                    img_mat[y_end-split_disp_size[1]:y_end, x_start:x_end, 2] = c[2]


plt.imshow(img_mat)
plt.show()
img = Image.fromarray(img_mat)
#img.save("output_image.png")