import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import yaml




# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

model_labels_path = 'Output/mlp_test_negri_28-10-2024_14:49/labels/labeled_epoch_59.npy'
label_path = cfg['valid_path']
all_classes = cfg['class_enum']
train_indeces_npy = cfg['train_indeces']
train_indeces = np.load(train_indeces_npy)
npy = 'Output/mlp_test_negri_28-10-2024_14:49/losses/epoch_59_valid_losses.npy'

data = np.load(npy, allow_pickle=True)
print("DATA", min(data))

true_dataset = np.load(label_path, allow_pickle=True)
true_dataset_info = true_dataset[0]
true_dataset_coords1 = true_dataset[1]

dataset_labeled = true_dataset_coords1[true_dataset_coords1[:,0] != -1]

test_indeces = np.setdiff1d(range(np.array(dataset_labeled.shape[0])), train_indeces)
true_dataset_coords = []
for i in test_indeces:
    true_dataset_coords.append(true_dataset_coords1[[i]])

print(true_dataset_coords)
dataset = np.load(model_labels_path, allow_pickle=True)
dataset_info = dataset[0]
dataset_coords = dataset[1]
acc = 0
total = 0


for i in range(len(dataset_coords)):
    for j in range(len(true_dataset_coords)):
        if dataset_coords[i][6] == true_dataset_coords[j][6]:
            if dataset_coords[i][0] == true_dataset_coords[j][0] and dataset_coords[i][1] == true_dataset_coords[j][1] and int(true_dataset_coords[j][4]) != -1:
                
                # Extract the label
                label = int(dataset_coords[i][4])
                true_label = int(true_dataset_coords[j][4])

                
                # Check if the predicted label matches the true label
                if label == true_label:
                    acc += 1  # Increment accuracy for the label
                
                total += 1  # Increment total count for the label
                # Break after finding the match for this entry to avoid unnecessary iterations
                break
totacc = (acc/total) * 100
print("Accuracy:", totacc)

