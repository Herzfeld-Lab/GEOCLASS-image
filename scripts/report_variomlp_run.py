#!/usr/bin/env python
"""Create a metrics report and presentation figures for one VarioMLP run."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


PALETTE = {
    "train": "#2b6cb0",
    "valid": "#c2410c",
    "accuracy": "#2f855a",
    "macro_f1": "#805ad5",
    "macro_precision": "#d69e2e",
    "macro_recall": "#319795",
}

METRIC_MARKERS = {
    "accuracy": "o",
    "macro_f1": "s",
    "macro_precision": "^",
    "macro_recall": "D",
}


def sort_label_dir(name):
    return int(name) if name.isdigit() else name


def epoch_from_path(path):
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_loss_series(run_dir):
    train_files = sorted((run_dir / "losses").glob("*_train_losses.npy"))
    valid_files = sorted((run_dir / "losses").glob("*_valid_losses.npy"))
    if not train_files or not valid_files:
        return None, None

    train_file = max(train_files, key=lambda path: epoch_from_path(path) or -1)
    valid_file = max(valid_files, key=lambda path: epoch_from_path(path) or -1)
    train_losses = np.asarray(np.load(train_file), dtype=float).reshape(-1)
    valid_losses = np.asarray(np.load(valid_file), dtype=float).reshape(-1)
    return train_losses, valid_losses


def label_files(run_dir):
    files = sorted((run_dir / "labels").glob("labeled_epoch_*.npy"))
    return sorted(files, key=lambda path: epoch_from_path(path) or -1)


def class_names(cfg, num_classes):
    names = cfg.get("class_enum")
    if not names:
        return [str(i) for i in range(num_classes)]
    return [names[i] if i < len(names) else str(i) for i in range(num_classes)]


def reconstruct_heldout_row_ids(cfg):
    training_folder = Path(cfg["training_img_path"])
    train_indices_path = Path(cfg["train_indices"])
    if not training_folder.exists() or not train_indices_path.exists():
        return None, {}

    train_indices = set(np.load(train_indices_path).astype(int).tolist())
    image_paths = []
    for label_name in sorted(os.listdir(training_folder), key=sort_label_dir):
        label_path = training_folder / label_name
        if not label_path.is_dir():
            continue
        for image_name in sorted(os.listdir(label_path)):
            if image_name.lower().endswith(".png"):
                image_paths.append(label_path / image_name)

    heldout_positions = [i for i in range(len(image_paths)) if i not in train_indices]
    row_ids = []
    for position in heldout_positions:
        match = re.search(r"_(\d+)\.png$", image_paths[position].name)
        if not match:
            raise ValueError(f"Cannot parse dataset row id from {image_paths[position]}")
        row_ids.append(int(match.group(1)))

    return np.asarray(row_ids, dtype=int), {
        "truth_source": "heldout_training_folder_split",
        "training_images": len(image_paths),
        "train_indices": len(train_indices),
    }


def truth_from_valid_path(cfg, dataset_rows):
    valid_path = Path(str(cfg.get("valid_path", "")))
    if not valid_path.exists():
        return None, {}

    valid_rows = np.load(valid_path, allow_pickle=True)[1]
    row_by_key = {
        (row[0], row[1], row[6]): int(row[4])
        for row in dataset_rows
        if int(row[4]) >= 0
    }
    row_ids = []
    labels = []
    dataset_key_to_index = {
        (row[0], row[1], row[6]): index for index, row in enumerate(dataset_rows)
    }
    for row in valid_rows:
        key = (row[0], row[1], row[6])
        if key in dataset_key_to_index and key in row_by_key and int(row[4]) >= 0:
            row_ids.append(dataset_key_to_index[key])
            labels.append(int(row[4]))

    if not row_ids:
        return None, {}
    return np.asarray(row_ids, dtype=int), {
        "truth_source": "valid_path",
        "valid_path": str(valid_path),
        "truth_labels_override": labels,
    }


def select_truth_rows(cfg, dataset_rows, mode):
    if mode in ("auto", "heldout-folder"):
        row_ids, meta = reconstruct_heldout_row_ids(cfg)
        if row_ids is not None or mode == "heldout-folder":
            return row_ids, None, meta

    if mode in ("auto", "valid-path"):
        row_ids, meta = truth_from_valid_path(cfg, dataset_rows)
        if row_ids is not None or mode == "valid-path":
            labels_override = meta.pop("truth_labels_override", None)
            return row_ids, labels_override, meta

    row_ids = np.where(dataset_rows[:, 4].astype(int) >= 0)[0]
    return row_ids, None, {"truth_source": "all_labeled_rows"}


def evaluate_label_file(label_file, dataset_rows, row_ids, labels_override, num_classes, confidence_threshold):
    predictions = np.load(label_file, allow_pickle=True)[1]
    y_true = (
        np.asarray(labels_override, dtype=int)
        if labels_override is not None
        else dataset_rows[row_ids, 4].astype(int)
    )
    y_pred = predictions[row_ids, 4].astype(int)
    confidence = predictions[row_ids, 5].astype(float)

    valid = (
        (y_true >= 0)
        & (y_pred >= 0)
        & (y_pred < num_classes)
        & (confidence >= confidence_threshold)
    )
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    confidence = confidence[valid]
    labels = list(range(num_classes))

    if len(y_true) == 0:
        raise ValueError(f"No rows left after filtering {label_file}")

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    total_rows = len(row_ids)
    epoch = epoch_from_path(label_file)
    return {
        "epoch": epoch,
        "label_file": str(label_file),
        "n_evaluated": int(len(y_true)),
        "coverage": float(len(y_true) / total_rows) if total_rows else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "avg_confidence": float(np.mean(confidence)),
        "per_class": [
            {
                "class": i,
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(per_support[i]),
            }
            for i in labels
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def metric_rows(metrics_by_epoch):
    keys = [
        "epoch",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "avg_confidence",
        "coverage",
        "n_evaluated",
        "label_file",
    ]
    return [{key: metric.get(key) for key in keys} for metric in metrics_by_epoch]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_per_class_csv(path, metrics, names):
    rows = []
    for item in metrics["per_class"]:
        rows.append(
            {
                "class": item["class"],
                "class_name": names[item["class"]],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
                "support": item["support"],
            }
        )
    write_csv(path, rows)


def set_presentation_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.titlesize": 16,
            "legend.fontsize": 11,
        }
    )


def plot_loss_and_metrics(path, train_losses, valid_losses, metrics_by_epoch, title):
    fig, ax_loss = plt.subplots(figsize=(11, 6.5))
    epochs = np.arange(len(train_losses))
    ax_loss.plot(epochs, train_losses, color=PALETTE["train"], linewidth=2.3, label="Training loss")
    ax_loss.plot(epochs, valid_losses, color=PALETTE["valid"], linewidth=2.3, label="Validation loss")
    best_epoch = int(np.nanargmin(valid_losses))
    ax_loss.axvline(best_epoch, color="#4a5568", linestyle="--", linewidth=1.2, alpha=0.75)
    ax_loss.scatter([best_epoch], [valid_losses[best_epoch]], color=PALETTE["valid"], s=70, zorder=5)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(title)

    ax_metric = ax_loss.twinx()
    metric_keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    metric_labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "macro_precision": "Macro precision",
        "macro_recall": "Macro recall",
    }
    # metric_offsets = np.linspace(-0.9, 0.9, len(metric_keys)) if len(metrics_by_epoch) == 1 else np.zeros(len(metric_keys))
    # for key in metric_keys:
    #     key_index = metric_keys.index(key)
    #     xs = [metric["epoch"] + metric_offsets[key_index] for metric in metrics_by_epoch if metric.get("epoch") is not None]
    #     ys = [metric[key] for metric in metrics_by_epoch if metric.get("epoch") is not None]
    #     if not xs:
    #         continue
    #     if len(xs) == 1:
    #         ax_metric.scatter(
    #             xs,
    #             ys,
    #             marker=METRIC_MARKERS[key],
    #             s=95,
    #             edgecolor="white",
    #             linewidth=0.9,
    #             zorder=6,
    #             color=PALETTE[key],
    #             label=metric_labels[key],
    #         )
    #         continue

    #     ax_metric.plot(
    #         xs,
    #         ys,
    #         marker=METRIC_MARKERS[key],
    #         markersize=7,
    #         linewidth=2.0,
    #         color=PALETTE[key],
    #         label=metric_labels[key],
    #     )
    # ax_metric.set_ylim(0, 1)
    # ax_metric.set_ylabel("Classification metric")

    handles_1, labels_1 = ax_loss.get_legend_handles_labels()
    handles_2, labels_2 = ax_metric.get_legend_handles_labels()
    # ax_loss.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right", frameon=True)
    ax_loss.legend(handles_1, labels_1, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(path, metrics, title):
    labels = ["Accuracy", "Macro F1", "Macro precision", "Macro recall", "Weighted F1"]
    values = [
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["macro_precision"],
        metrics["macro_recall"],
        metrics["weighted_f1"],
    ]
    colors = [
        PALETTE["accuracy"],
        PALETTE["macro_f1"],
        PALETTE["macro_precision"],
        PALETTE["macro_recall"],
        "#718096",
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=11)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(path, metrics, names, title):
    labels = [f"{item['class']}: {names[item['class']]}" for item in metrics["per_class"]]
    y = np.arange(len(labels))
    height = 0.24
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(y - height, [item["precision"] for item in metrics["per_class"]], height, label="Precision", color=PALETTE["macro_precision"])
    ax.barh(y, [item["recall"] for item in metrics["per_class"]], height, label="Recall", color=PALETTE["macro_recall"])
    ax.barh(y + height, [item["f1"] for item in metrics["per_class"]], height, label="F1", color=PALETTE["macro_f1"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(path, metrics, names, title, normalize):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    annot = matrix.astype(int)
    values = matrix
    fmt = "d"
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        values = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        annot = np.asarray([[f"{value:.0%}" for value in row] for row in values])
        fmt = ""

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        values,
        annot=annot,
        fmt=fmt,
        cmap="YlGnBu",
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Share" if normalize else "Count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, run_dir, metrics, loss_summary, truth_meta, output_files):
    lines = [
        f"# VarioMLP Run Report: `{run_dir.name}`",
        "",
        "## Summary",
        "",
        f"- Best validation loss: `{loss_summary['best_val_loss']:.6f}` at epoch `{loss_summary['best_epoch']}`",
        f"- Final training loss: `{loss_summary['final_train_loss']:.6f}`",
        f"- Final validation loss: `{loss_summary['final_val_loss']:.6f}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Macro F1: `{metrics['macro_f1']:.4f}`",
        f"- Macro precision: `{metrics['macro_precision']:.4f}`",
        f"- Macro recall: `{metrics['macro_recall']:.4f}`",
        f"- Weighted F1: `{metrics['weighted_f1']:.4f}`",
        f"- Average confidence: `{metrics['avg_confidence']:.4f}`",
        f"- Evaluated rows: `{metrics['n_evaluated']}`",
        f"- Truth source: `{truth_meta.get('truth_source')}`",
        "",
        "## Figures",
        "",
    ]
    for label, file_path in output_files.items():
        lines.append(f"- {label}: `{file_path}`")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--truth-mode",
        choices=["auto", "heldout-folder", "valid-path", "all-labeled"],
        default="auto",
        help="How to choose true labels for metrics. Default reconstructs the training held-out split.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    args = parser.parse_args()

    cfg = yaml.load(args.config.read_text(), Loader=yaml.FullLoader)
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "metrics_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = np.load(cfg["npy_path"], allow_pickle=True)[1]
    num_classes = int(cfg["num_classes"])
    names = class_names(cfg, num_classes)
    row_ids, labels_override, truth_meta = select_truth_rows(cfg, dataset_rows, args.truth_mode)
    if row_ids is None or len(row_ids) == 0:
        raise SystemExit("No truth rows found for the selected truth mode.")

    train_losses, valid_losses = load_loss_series(run_dir)
    if train_losses is None or valid_losses is None:
        raise SystemExit(f"No train/validation loss arrays found under {run_dir / 'losses'}")

    labels = label_files(run_dir)
    if not labels:
        raise SystemExit(f"No prediction label files found under {run_dir / 'labels'}")

    metrics_by_epoch = [
        evaluate_label_file(
            label_file,
            dataset_rows,
            row_ids,
            labels_override,
            num_classes,
            args.confidence_threshold,
        )
        for label_file in labels
    ]
    summary_metrics = max(metrics_by_epoch, key=lambda item: item["macro_f1"])
    best_epoch = int(np.nanargmin(valid_losses))
    loss_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": float(valid_losses[best_epoch]),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(valid_losses[-1]),
    }

    set_presentation_style()
    figures = {
        "Loss and metrics": output_dir / "loss_and_metrics.png",
        "Summary metrics": output_dir / "summary_metrics.png",
        "Per-class metrics": output_dir / "per_class_metrics.png",
        "Confusion matrix": output_dir / "confusion_matrix.png",
        "Normalized confusion matrix": output_dir / "confusion_matrix_normalized.png",
    }
    plot_loss_and_metrics(
        figures["Loss and metrics"],
        train_losses,
        valid_losses,
        metrics_by_epoch,
        f"{run_dir.name}: loss and held-out metrics",
    )
    plot_metric_summary(
        figures["Summary metrics"],
        summary_metrics,
        f"{run_dir.name}: summary metrics at epoch {summary_metrics['epoch']}",
    )
    plot_per_class(
        figures["Per-class metrics"],
        summary_metrics,
        names,
        f"{run_dir.name}: per-class metrics",
    )
    plot_confusion_matrix(
        figures["Confusion matrix"],
        summary_metrics,
        names,
        f"{run_dir.name}: confusion matrix",
        normalize=False,
    )
    plot_confusion_matrix(
        figures["Normalized confusion matrix"],
        summary_metrics,
        names,
        f"{run_dir.name}: normalized confusion matrix",
        normalize=True,
    )

    metrics_json = output_dir / "metrics_report.json"
    metrics_csv = output_dir / "metrics_by_epoch.csv"
    per_class_csv = output_dir / "per_class_metrics.csv"
    markdown = output_dir / "README.md"

    report = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "config": str(args.config),
        "confidence_threshold": args.confidence_threshold,
        "truth": truth_meta,
        "loss_summary": loss_summary,
        "selected_metrics_epoch": summary_metrics["epoch"],
        "metrics_by_epoch": metrics_by_epoch,
        "figures": {key: str(value) for key, value in figures.items()},
    }
    metrics_json.write_text(json.dumps(report, indent=2))
    write_csv(metrics_csv, metric_rows(metrics_by_epoch))
    write_per_class_csv(per_class_csv, summary_metrics, names)
    write_markdown(markdown, run_dir, summary_metrics, loss_summary, truth_meta, figures)

    print(f"Report written to {output_dir}")
    print(f"Selected metrics epoch: {summary_metrics['epoch']}")
    print(
        "accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} "
        "macro_precision={macro_precision:.4f} macro_recall={macro_recall:.4f}".format(
            **summary_metrics
        )
    )
    print(f"best_val_loss={loss_summary['best_val_loss']:.6f} at epoch {loss_summary['best_epoch']}")


if __name__ == "__main__":
    main()
