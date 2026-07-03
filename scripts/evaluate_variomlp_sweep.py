#!/usr/bin/env python
"""Summarize VarioMLP sweep runs with validation loss and held-out metrics."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def sort_label_dir(name):
    return int(name) if name.isdigit() else name


def best_validation_loss(run_dir):
    loss_files = sorted((run_dir / "losses").glob("*_valid_losses.npy"))
    if not loss_files:
        return None, None, None

    def epoch_from_name(path):
        match = re.search(r"epoch_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    loss_file = max(loss_files, key=epoch_from_name)
    losses = np.asarray(np.load(loss_file), dtype=float).reshape(-1)
    best_epoch = int(np.nanargmin(losses))
    return best_epoch, float(losses[best_epoch]), float(losses[-1])


def reconstruct_heldout_row_ids(cfg):
    training_folder = Path(cfg["training_img_path"])
    train_indices = set(np.load(cfg["train_indices"]).astype(int).tolist())
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

    return np.asarray(row_ids, dtype=int), len(image_paths), len(train_indices)


def latest_label_file(run_dir):
    label_files = sorted((run_dir / "labels").glob("labeled_*.npy"))
    if not label_files:
        return None

    def epoch_from_name(path):
        match = re.search(r"labeled_epoch_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return max(label_files, key=epoch_from_name)


def run_metrics(run_dir, dataset_rows, heldout_row_ids, num_classes):
    label_file = latest_label_file(run_dir)
    if label_file is None:
        return None

    predictions = np.load(label_file, allow_pickle=True)[1]
    y_true = dataset_rows[heldout_row_ids, 4].astype(int)
    y_pred = predictions[heldout_row_ids, 4].astype(int)
    confidence = predictions[heldout_row_ids, 5].astype(float)

    valid = (y_true >= 0) & (y_pred >= 0) & (y_pred < num_classes)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    confidence = confidence[valid]
    labels = list(range(num_classes))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    return {
        "label_file": str(label_file),
        "n_heldout": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "avg_confidence": float(np.mean(confidence)) if len(confidence) else None,
    }


def summarize_run(run_dir, dataset_rows, heldout_row_ids, num_classes):
    best_epoch, best_val_loss, final_val_loss = best_validation_loss(run_dir)
    metrics = run_metrics(run_dir, dataset_rows, heldout_row_ids, num_classes)
    row = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
    }
    if metrics:
        row.update(metrics)
    return row


def write_csv(path, rows):
    fieldnames = [
        "run",
        "best_epoch",
        "best_val_loss",
        "final_val_loss",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "avg_confidence",
        "n_heldout",
        "label_file",
        "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.load(args.config.read_text(), Loader=yaml.FullLoader)
    dataset_rows = np.load(cfg["npy_path"], allow_pickle=True)[1]
    heldout_row_ids, image_count, train_count = reconstruct_heldout_row_ids(cfg)
    num_classes = int(cfg["num_classes"])

    run_dirs = sorted(path for path in args.sweep_dir.glob("mlp_*") if path.is_dir())
    rows = [
        summarize_run(run_dir, dataset_rows, heldout_row_ids, num_classes)
        for run_dir in run_dirs
    ]

    rows_by_macro_f1 = sorted(rows, key=lambda row: row.get("macro_f1", -1), reverse=True)
    rows_by_loss = sorted(
        rows,
        key=lambda row: row["best_val_loss"] if row["best_val_loss"] is not None else float("inf"),
    )

    print(
        f"Held-out rows: {len(heldout_row_ids)} "
        f"(training images: {image_count}, train indices: {train_count})"
    )
    print("\nRank by macro F1:")
    for row in rows_by_macro_f1:
        print(
            "{run}: macro_f1={macro_f1:.4f} accuracy={accuracy:.4f} "
            "weighted_f1={weighted_f1:.4f} best_val={best_val_loss:.6f} "
            "epoch={best_epoch}".format(**row)
        )

    print("\nRank by validation loss:")
    for row in rows_by_loss:
        print(
            "{run}: best_val={best_val_loss:.6f} epoch={best_epoch} "
            "macro_f1={macro_f1:.4f} accuracy={accuracy:.4f}".format(**row)
        )

    csv_path = args.csv or args.sweep_dir / "metrics_summary.csv"
    json_path = args.json or args.sweep_dir / "metrics_summary.json"
    write_csv(csv_path, rows_by_macro_f1)
    json_path.write_text(json.dumps(rows_by_macro_f1, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
