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
        if dataset_coords[i][6] == true_dataset_coords[j][6]:
            if dataset_coords[i][0] == true_dataset_coords[j][0] and dataset_coords[i][1] == true_dataset_coords[j][1] and int(true_dataset_coords[j][4]) != -1:
                
                # Extract the label
                label = int(dataset_coords[i][4])
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
                confidence_dict[label] += int(dataset_coords[i][5] * 100)

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


model_labels_path1 = cfg['train_path1']
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

for i in range(len(dataset_coords1)):
    for j in range(len(true_dataset_coords1)):
        if dataset_coords1[i][6] == true_dataset_coords1[j][6]:
            if dataset_coords1[i][0] == true_dataset_coords1[j][0] and dataset_coords1[i][1] == true_dataset_coords1[j][1] and int(true_dataset_coords1[j][4]) != -1:
                
                # Extract the label
                label1 = int(dataset_coords1[i][4])
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
                confidence_dict1[label1] += int(dataset_coords1[i][5] * 100)

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

if len(accuracy_list1) < len(accuracy_list):
    accuracy_list1 += [0] * (len(accuracy_list) - len(accuracy_list1))

if len(confidence_list1) < len(confidence_list):
    confidence_list1 += [0] * (len(confidence_list) - len(confidence_list1))

x_ticks = np.arange(min(labels_list), max(labels_list) + 1)

# Define bar width
bar_width = 0.35
# Create positions for the bars (with a slight shift for the second set of bars)
r1 = np.arange(len(labels_list))  # Positions for the first set of bars
r2 = [x + bar_width for x in r1]  # Positions for the second set of bars

# Plot histogram for accuracy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(r1, accuracy_list, color='skyblue', width=bar_width, label='53 Lag steps')
plt.bar(r2, accuracy_list1, color='green', width=bar_width, label='18 Lag steps')
plt.xlabel('Labels')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Class')
plt.xticks([r + bar_width / 2 for r in r1], labels_list)  # Adjust x-ticks to be in the middle of the two bars
plt.legend()

# Plot histogram for average confidence
plt.subplot(1, 2, 2)
plt.bar(r1, confidence_list, color='salmon', width=bar_width, label='53 Lag steps')
plt.bar(r2, confidence_list1, color='purple', width=bar_width, label='18 Lag steps')
plt.xlabel('Labels')
plt.ylabel('Average Confidence')
plt.title('Average Confidence per Class')
plt.xticks([r + bar_width / 2 for r in r1], labels_list)  # Adjust x-ticks to be in the middle of the two bars
plt.legend()

plt.tight_layout()
plt.show()



