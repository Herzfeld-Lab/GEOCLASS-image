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
accuracy_dict = {}
confidence_dict = {}
total_dict = {}

for i in range(len(dataset_coords)):
    for j in range(len(true_dataset_coords)):
            if dataset_coords[i][0] == true_dataset_coords[j][0] and dataset_coords[i][1] == true_dataset_coords[j][1] and int(true_dataset_coords[j][2]) != -1:
                
                # Extract the label
                label = int(dataset_coords[i][2])
                true_label = int(true_dataset_coords[j][2])

                # Initialize dictionary entries if the label doesn't exist yet
                if label not in accuracy_dict:
                    accuracy_dict[label] = 0
                    confidence_dict[label] = 0
                    total_dict[label] = 0

                # Append to the list of labels and true_labels
                labels.append(label)
                true_labels.append(true_label)
                
                # Check if the predicted label matches the true label
                if label == true_label:
                    accuracy_dict[label] += 1  # Increment accuracy for the label
                
                total_dict[label] += 1  # Increment total count for the label
                
                # Add the confidence value (scaled) for this label
                confidence_dict[label] += int(dataset_coords[i][3] * 100)

                # Break after finding the match for this entry to avoid unnecessary iterations
                break

labels_list = []
accuracy_list = []
confidence_list = []

# Display the results for each label
for label in accuracy_dict:
    total = total_dict[label]
    accuracy = accuracy_dict[label]
    confidence = confidence_dict[label]

    # Calculate percentage accuracy and average confidence
    accuracy_percentage = (accuracy / total) * 100 if total > 0 else 0
    average_confidence = confidence / total if total > 0 else 0
    
    # Store results for plotting
    labels_list.append(label)
    accuracy_list.append(accuracy_percentage)
    confidence_list.append(average_confidence)

# Plot histogram for accuracy
x_ticks = np.arange(min(labels_list), max(labels_list) + 1)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(labels_list, accuracy_list, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Class')
plt.xticks(x_ticks)

# Plot histogram for average confidence
plt.subplot(1, 2, 2)
plt.bar(labels_list, confidence_list, color='salmon')
plt.xlabel('Labels')
plt.ylabel('Average Confidence')
plt.title('Average Confidence per Class')
plt.xticks(x_ticks)

plt.tight_layout()
plt.show()

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

