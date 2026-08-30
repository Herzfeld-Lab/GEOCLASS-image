import argparse
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import yaml
from pathlib import Path
import os


# Parse command line flags and normalize paths for cross-platform
def parse_args():
    p = argparse.ArgumentParser(description='Plot per-class MS metrics')
    p.add_argument('config', type=str, help='YAML config file')
    p.add_argument('--comp_labels', type=str, default=None, help='Optional comparison labels .npy file')
    p.add_argument('--comp_name', type=str, default='Comp', help='Comparison model name')
    p.add_argument('--save_metrics', type=str, default=None, help='CSV output path to save metrics')
    p.add_argument('--no_plot', action='store_true', help='Do not show plots')
    return p.parse_args()


def resolve_path(value, base_dir=None):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.lower() in {'none', 'null', 'nil'}:
            return None
    p = Path(str(value)).expanduser()
    # Keep paths exactly as configured; do not prepend the config directory.
    # The project config files are already written with the intended location.
    return p


def load_config(path):
    cfg_path = Path(path).expanduser()
    with open(cfg_path, 'r') as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.FullLoader)


args = parse_args()
cfg = load_config(args.config)

# Main model labels path comes from config; make it platform-safe without rewriting it relative to the config file.
model_labels_path = resolve_path(cfg['train_path'])
label_path = resolve_path(cfg['valid_path'])
train_indices_npy = resolve_path(cfg['train_indices'])
all_classes = cfg.get('class_enum_MS', cfg.get('class_enum'))

pred = np.load(model_labels_path, allow_pickle=True)
true = np.load(label_path, allow_pickle=True)

print(pred[1].shape)
print(true[1].shape)

p = pred[1]
t = true[1]

# For MS split rows, the label/value columns are:
# [pan_x, pan_y, ms_x, ms_y, utm_x, utm_y, pan_label, pan_conf, ms_label, ms_conf, img_num]
# So the class label is column 8, not 4.
LABEL_COL = 8
CONF_COL = 9

print("pred unique labels:", np.unique(p[:, LABEL_COL]))
print("true unique labels:", np.unique(t[:, LABEL_COL]))

# Optional comparison labels provided via CLI override hardcoded path
comp_labels_path = resolve_path(args.comp_labels) if args.comp_labels else None

if label_path is None:
    raise FileNotFoundError("The config field 'valid_path' is missing/None. Set it to the ground-truth label .npy file or use a valid config.")

true_dataset = np.load(str(label_path), allow_pickle=True)
true_dataset_info = true_dataset[0]
true_dataset_coords = true_dataset[1]


dataset = np.load(str(model_labels_path), allow_pickle=True)
dataset_info = dataset[0]
dataset_coords = dataset[1]


def map_preds_to_truth(pred_coords, true_dataset_coords, test_indeces):
    y_true = []
    y_pred = []
    confs = []
    # build a list of test coords from indices
    test_coords = [pred_coords[i] for i in test_indeces]

    for tc in test_coords:
        for j in range(len(true_dataset_coords)):
            # match by filename/index and coordinates
            if tc[6] == true_dataset_coords[j][6]:
                if tc[0] == true_dataset_coords[j][0] and tc[1] == true_dataset_coords[j][1] and int(true_dataset_coords[j][LABEL_COL]) != -1:
                    y_pred.append(int(tc[LABEL_COL]))
                    y_true.append(int(true_dataset_coords[j][LABEL_COL]))
                    confs.append(float(tc[CONF_COL]))
                    break
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int), np.array(confs, dtype=float)
train_indices = np.load(str(train_indices_npy))
test_indeces = np.setdiff1d(range(np.array(dataset_coords.shape[0])), train_indices)

# Map predictions to ground truth for the main model
y_true, y_pred, confs = map_preds_to_truth(dataset_coords, true_dataset_coords, test_indeces)

num_classes = len(all_classes)

def compute_class_metrics(y_true, y_pred, num_classes):
    # Ensure labels present in range
    labels_range = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)

    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        accuracy_per_class = (tp + tn) / cm.sum() if cm.sum() > 0 else np.zeros_like(tp)

    overall_acc = accuracy_score(y_true, y_pred) if y_true.size > 0 else 0.0
    return {
        'confusion_matrix': cm,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'recall': recall,
        'precision': precision,
        'accuracy_per_class': accuracy_per_class,
        'overall_accuracy': overall_acc,
        'support': cm.sum(axis=1)
    }


metrics_comp = None

# If comparison labels path provided and exists, load and compute
if comp_labels_path and comp_labels_path.exists():
    dataset1 = np.load(str(comp_labels_path), allow_pickle=True)
    dataset_info1 = dataset1[0]
    dataset_coords1 = dataset1[1]
    y_true1, y_pred1, confs1 = map_preds_to_truth(dataset_coords1, true_dataset_coords, test_indeces)
    metrics_comp = compute_class_metrics(y_true1, y_pred1, num_classes)

# Compute metrics for main model
metrics_main = compute_class_metrics(y_true, y_pred, num_classes)

# Print summary
print(f"Overall accuracy (main): {metrics_main['overall_accuracy']*100:.2f}%")
if metrics_comp is not None:
    print(f"Overall accuracy ({args.comp_name}): {metrics_comp['overall_accuracy']*100:.2f}%")

# Prepare plotting data (per-class recall and precision)
labels_list = list(range(num_classes))
recall_list = (metrics_main['recall'] * 100).tolist()
precision_list = (metrics_main['precision'] * 100).tolist()
if metrics_comp is not None:
    recall_list1 = (metrics_comp['recall'] * 100).tolist()
    precision_list1 = (metrics_comp['precision'] * 100).tolist()
else:
    recall_list1 = [0] * len(labels_list)
    precision_list1 = [0] * len(labels_list)

# Plot recall and precision comparisons side-by-side
bar_width = 0.35
r = np.arange(len(labels_list))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Recall subplot
axes[0].bar(r, recall_list, color='salmon', width=bar_width, label='Main')
if metrics_comp is not None:
    axes[0].bar(r + bar_width, recall_list1, color='purple', width=bar_width, label=args.comp_name)
axes[0].set_xticks(r + bar_width / 2 if metrics_comp is not None else r)
axes[0].set_xticklabels(labels_list)
axes[0].set_xlabel('Labels')
axes[0].set_ylabel('Recall (%)')
axes[0].legend() if metrics_comp is not None else None
axes[0].set_title('Per-class Recall')

# Precision subplot
axes[1].bar(r, precision_list, color='salmon', width=bar_width, label='Main')
if metrics_comp is not None:
    axes[1].bar(r + bar_width, precision_list1, color='purple', width=bar_width, label=args.comp_name)
axes[1].set_xticks(r + bar_width / 2 if metrics_comp is not None else r)
axes[1].set_xticklabels(labels_list)
axes[1].set_xlabel('Labels')
axes[1].set_ylabel('Precision (%)')
axes[1].legend() if metrics_comp is not None else None
axes[1].set_title('Per-class Precision')

plt.tight_layout()
if not args.no_plot:
    plt.show()
