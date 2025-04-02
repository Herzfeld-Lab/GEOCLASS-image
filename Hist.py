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
train_indices_npy = cfg['train_indices']
train_indices = np.load(train_indices_npy)
test_indeces = np.setdiff1d(range(np.array(dataset_coords.shape[0])), train_indices)

test_coords = []
test_coords1 = []
for i in test_indeces:
    test_coords.append(dataset_coords[i])

for i in range(len(test_coords)):
    for j in range(len(true_dataset_coords)):
        if test_coords[i][6] == true_dataset_coords[j][6]:
            if test_coords[i][0] == true_dataset_coords[j][0] and test_coords[i][1] == true_dataset_coords[j][1] and int(true_dataset_coords[j][4]) != -1:
                #print(i)
                # Extract the label
                label = int(test_coords[i][4])
                true_label = int(true_dataset_coords[j][4])

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
                confidence_dict[label] += int(test_coords[i][5] * 100)

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

sorted_indices = np.argsort(labels_list)
labels_list = np.array(labels_list)[sorted_indices].tolist()
accuracy_list = np.array(accuracy_list)[sorted_indices].tolist()
confidence_list = np.array(confidence_list)[sorted_indices].tolist()


model_labels_path1 = 'Output/12_class_runs/Resnet18/mlp_12_negri_01-04-2025_10:20/labels/labeled_epoch_45.npy'
label_path1 = cfg['valid_path']

true_dataset1 = np.load(label_path1, allow_pickle=True)
true_dataset_info1 = true_dataset1[0]
true_dataset_coords1 = true_dataset1[1]


dataset1 = np.load(model_labels_path1, allow_pickle=True)
dataset_info1 = dataset1[0]
dataset_coords1 = dataset1[1]

true_labels1 = []
labels1 = []
accuracy_dict1 = {}
confidence_dict1 = {}
total_dict1 = {}
for i in test_indeces:
    test_coords1.append(dataset_coords1[i])

for i in range(len(test_coords1)):
    for j in range(len(true_dataset_coords1)):
        if test_coords1[i][6] == true_dataset_coords1[j][6]:
            if test_coords1[i][0] == true_dataset_coords1[j][0] and test_coords1[i][1] == true_dataset_coords1[j][1] and int(true_dataset_coords1[j][4]) != -1:
                #print(i)
                # Extract the label
                label1 = int(test_coords1[i][4])
                true_label1 = int(true_dataset_coords1[j][4])

                # Initialize dictionary entries if the label doesn't exist yet
                if label1 not in accuracy_dict1:
                    accuracy_dict1[label1] = 0
                    confidence_dict1[label1] = 0
                    total_dict1[label1] = 0

                # Append to the list of labels and true_labels
                labels1.append(label1)
                true_labels1.append(true_label1)
                
                # Check if the predicted label matches the true label
                if label1 == true_label1:
                    accuracy_dict1[label1] += 1  # Increment accuracy for the label
                
                total_dict1[label1] += 1  # Increment total count for the label
                
                # Add the confidence value (scaled) for this label
                confidence_dict1[label1] += int(test_coords1[i][5] * 100)

                # Break after finding the match for this entry to avoid unnecessary iterations
                break

labels_list1 = []
accuracy_list1 = []
confidence_list1 = []

# Display the results for each label
for label1 in accuracy_dict1:
    total1 = total_dict1[label1]
    accuracy1 = accuracy_dict1[label1]
    confidence1 = confidence_dict1[label1]

    # Calculate percentage accuracy and average confidence
    accuracy_percentage1 = (accuracy1 / total1) * 100 if total1 > 0 else 0
    average_confidence1 = confidence1 / total1 if total1 > 0 else 0
    
    # Store results for plotting
    labels_list1.append(label1)
    accuracy_list1.append(accuracy_percentage1)
    confidence_list1.append(average_confidence1)

sorted_indices1 = np.argsort(labels_list1)
labels_list1 = np.array(labels_list1)[sorted_indices1].tolist()
accuracy_list1 = np.array(accuracy_list1)[sorted_indices1].tolist()
confidence_list1 = np.array(confidence_list1)[sorted_indices1].tolist()

if len(accuracy_list1) < len(accuracy_list):
    accuracy_list1 += [0] * (len(accuracy_list) - len(accuracy_list1))

if len(confidence_list1) < len(confidence_list):
    confidence_list1 += [0] * (len(confidence_list) - len(confidence_list1))

x_ticks = np.arange(min(labels_list), max(labels_list) + 1)
totaccuracy = 0
totaccuracy1 = 0
for i in accuracy_list:
    totaccuracy += i
for i in accuracy_list1:
    totaccuracy1 += i


print("Test Accuracy", totaccuracy/len(accuracy_list))
print("Origininal Accuracy", totaccuracy1/len(accuracy_list1))

bar_width = 0.35

# Ensure label sets are the same length
all_labels = sorted(set(labels_list) | set(labels_list1))

# Fill in missing values with 0
accuracy_dict = {label: 0 for label in all_labels}
accuracy_dict1 = {label: 0 for label in all_labels}

accuracy_dict.update(dict(zip(labels_list, accuracy_list)))
accuracy_dict1.update(dict(zip(labels_list1, accuracy_list1)))

accuracy_list = [accuracy_dict[label] for label in all_labels]
accuracy_list1 = [accuracy_dict1[label] for label in all_labels]

# X positions for bars
r1 = np.arange(len(all_labels))
r2 = r1 + bar_width  # Shift second set slightly

# Plot bars
plt.bar(r1, accuracy_list, color='salmon', width=bar_width, label='VarioMLP')
plt.bar(r2, accuracy_list1, color='purple', width=bar_width, label='ResNet18')

# Set x-ticks correctly
plt.xticks(r1 + bar_width / 2, all_labels)  # Center tick labels

plt.xlabel("Labels")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()
