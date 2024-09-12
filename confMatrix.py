import argparse
import seaborn as sns
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

model_labels_path = cfg['train_path']
label_path = cfg['valid_path']
all_classes = cfg['class_enum']


true_dataset = np.load(label_path, allow_pickle=True)
true_dataset_info = true_dataset[0]
true_dataset_coords = true_dataset[1]


dataset = np.load(model_labels_path, allow_pickle=True)
dataset_info = dataset[0]
dataset_coords = dataset[1]

true_labels = []
labels = []



for i in range(len(dataset_coords)):
    for j in range(len(true_dataset_coords)):
        if dataset_coords[i][6] == true_dataset_coords[j][6]:
            if dataset_coords[i][0] == true_dataset_coords[j][0] and dataset_coords[i][1] == true_dataset_coords[j][1] and int(true_dataset_coords[j][4]) != -1:
                labels.append(int(dataset_coords[i][4]))
                true_labels.append(int(true_dataset_coords[j][4]))

active_classes = sorted(np.unique(np.concatenate((true_labels, labels))))

# Use these active class indices to filter class names
class_names = [all_classes[i] for i in active_classes]

conf_matrix = confusion_matrix(true_labels, labels)
# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

