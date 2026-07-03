#!/usr/bin/env python
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio as rio
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import generate_config_silas, get_img_sigma, scaleImage


def stratified_indices(labels, train_fraction, seed, num_classes):
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        if class_indices.size == 0:
            continue
        target = int(np.floor(train_fraction * class_indices.size))
        if class_indices.size > 1:
            target = min(max(1, target), class_indices.size - 1)
        else:
            target = 1
        selected.extend(rng.choice(class_indices, size=target, replace=False).tolist())

    target_total = int(round(train_fraction * labels.size))
    remaining = sorted(set(range(labels.size)) - set(selected))
    if len(selected) < target_total and remaining:
        selected.extend(rng.choice(remaining, size=min(target_total - len(selected), len(remaining)), replace=False).tolist())
    elif len(selected) > target_total:
        selected = sorted(selected)
        selected = rng.choice(selected, size=target_total, replace=False).tolist()

    return np.array(sorted(selected), dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        default="Config/greenland_enthalpy/jakobshavn-wv/jakobshavn-ice-stream-2018-2021-6class-labeling.config",
        nargs="?",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--no_update_config", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader)
    data = np.load(cfg["npy_path"], allow_pickle=True)
    info, coords = data[0], data[1]
    labels = coords[:, 4].astype(int)
    labeled_rows = np.where(labels >= 0)[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"Training_data/Greenland/Jakobshavn-wv-2018-2021-6class-synced-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_id in range(cfg["num_classes"]):
        (output_dir / str(class_id)).mkdir(parents=True, exist_ok=True)

    win = np.asarray(info["winsize_pix"]).astype(int)
    image_mats = []
    image_sigmas = []
    for filename in info["filename"]:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(
                f"Source TIFF is missing: {path}\n"
                "Mount the source imagery volume, then rerun this script."
            )
        ds = rio.open(path)
        mat = ds.read(1)
        image_mats.append(mat)
        image_sigmas.append(get_img_sigma(mat[::10, ::10]))

    counts = {class_id: 0 for class_id in range(cfg["num_classes"])}
    folder_labels = []
    for row_idx in labeled_rows:
        row = coords[row_idx]
        class_id = int(row[4])
        image_id = int(row[6])
        r, c = row[0:2].astype(int)
        chip = image_mats[image_id][r : r + win[0], c : c + win[1]]
        chip = scaleImage(chip, image_sigmas[image_id])
        Image.fromarray(chip).save(output_dir / str(class_id) / f"{class_id}_{row_idx:05d}.png")
        counts[class_id] += 1
        folder_labels.append(class_id)

    train_indices = stratified_indices(
        folder_labels,
        float(cfg["train_test_split"]),
        args.seed,
        cfg["num_classes"],
    )
    train_indices_path = cfg_path.with_name(
        f"{cfg_path.stem}_{cfg['num_classes']}_{len(train_indices)}train_indices.npy"
    )
    np.save(train_indices_path, train_indices)

    if not args.no_update_config:
        cfg["training_img_path"] = str(output_dir)
        cfg["train_indices"] = str(train_indices_path)
        cfg["training_img_npy"] = "None"
        cfg_path.write_text(generate_config_silas(cfg))

    print(f"Wrote {sum(counts.values())} PNGs to {output_dir}")
    print(f"Class counts: {counts}")
    print(f"Wrote train indices: {train_indices_path} ({len(train_indices)} train / {len(folder_labels) - len(train_indices)} valid)")
    if not args.no_update_config:
        print(f"Updated config: {cfg_path}")


if __name__ == "__main__":
    main()
