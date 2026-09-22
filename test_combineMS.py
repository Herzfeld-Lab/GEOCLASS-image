"""Create a high-confidence water-and-crevasse mask from PAN and MS results.

The two input files must be labelled datasets created by the PAN and MS test
scripts.  The returned table retains the original 11-column format.  Only a
patch where both classifiers pass the water-and-crevasse test retains its PAN
and MS predictions; all other PAN/MS labels are written as -1 (unlabelled).
"""

import argparse
import os

import numpy as np
import yaml


#Setup

DEFAULT_CREVASSE_CLASSES = [
    'Shear',
    'Parallel',
    'Parallel Shear',
    'Subordinate Shear',
    'Multigenerational',
    'Multidirectional',
    'Chaos',
]
DEFAULT_WATER_CLASSES = ['Water on Ice']

# Fusion settings used by test_combineMS.py.  A patch is kept only when its
# PAN and MS predictions both belong to these lists and exceed their thresholds.
#water_crevasse_pan_classes: ['Shear', 'Parallel', 'Parallel Shear', 'Subordinate Shear', 'Multigenerational', 'Multidirectional', 'Chaos']
#water_crevasse_ms_classes: ['Water on Ice']
DEFAULT_PAN_CONF = 0.1
DEFAULT_MS_CONF = 0.1
WATER_CREV_LABEL_NAME = 'Water in Crevasse'


def class_ids(class_names, requested_names, role):
    """Resolve configured class names and fail clearly on a typo."""
    ids = []
    for name in requested_names:
        if name not in class_names:
            available = ', '.join(class_names)
            raise ValueError(
                f"Unknown {role} class {name!r}. Available classes: {available}"
            )
        ids.append(class_names.index(name))
    return np.asarray(ids, dtype=int)


def aligned_ms_rows(pan_rows, ms_rows):
    """Return MS rows in PAN-row order, checking that patches correspond."""
    if pan_rows.shape[0] != ms_rows.shape[0]:
        raise ValueError(
            f"PAN has {pan_rows.shape[0]} rows but MS has {ms_rows.shape[0]} rows."
        )

    # Normal test outputs preserve split-table order.  This fast path also
    # ensures the geometry and source image match before fusing predictions.
    key_columns = (0, 1, 2, 3, 4, 5, 10)
    if np.array_equal(pan_rows[:, key_columns], ms_rows[:, key_columns]):
        return ms_rows

    # If a producer reordered rows, restore correspondence by UTM coordinate
    # and source image.  The data generator creates one patch per such key.
    def key(row):
        return (float(row[4]), float(row[5]), int(row[10]))

    ms_by_key = {}
    for row in ms_rows:
        row_key = key(row)
        if row_key in ms_by_key:
            raise ValueError(f"Duplicate MS patch key {row_key}; cannot align outputs safely.")
        ms_by_key[row_key] = row

    try:
        return np.asarray([ms_by_key[key(row)] for row in pan_rows])
    except KeyError as exc:
        raise ValueError(
            f"The PAN and MS outputs do not describe the same patches; missing key {exc.args[0]}."
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description='Mask PAN/MS predictions to high-confidence water-filled crevasses.'
    )
    parser.add_argument('config', help='YAML config containing PAN and MS class enumerations.')
    parser.add_argument('--pan-labels', required=True, help='Labelled PAN .npy output from test_MS.py.')
    parser.add_argument('--ms-labels', required=True, help='Labelled MS .npy output from test_MS.py.')
    parser.add_argument(
        '--output',
        default=None,
        help='Output .npy path (default: alongside the PAN labels).',
    )
    parser.add_argument('--pan-threshold', type=float, default=None)
    parser.add_argument('--ms-threshold', type=float, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        cfg = yaml.safe_load(config_file)

    pan_classes = cfg['class_enum_PAN']
    ms_classes = cfg['class_enum_MS']
    crevasse_names = cfg.get('water_crevasse_pan_classes', DEFAULT_CREVASSE_CLASSES)
    water_names = cfg.get('water_crevasse_ms_classes', DEFAULT_WATER_CLASSES)
    pan_threshold = DEFAULT_PAN_CONF if args.pan_threshold is None else args.pan_threshold
    ms_threshold = DEFAULT_MS_CONF if args.ms_threshold is None else args.ms_threshold

    for name, threshold in (('PAN', pan_threshold), ('MS', ms_threshold)):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f'{name} threshold must be between 0 and 1, got {threshold}.')

    crevasse_ids = class_ids(pan_classes, crevasse_names, 'PAN crevasse')
    water_ids = class_ids(ms_classes, water_names, 'MS water')
    combined_label_name = WATER_CREV_LABEL_NAME

    pan_dataset = np.load(args.pan_labels, allow_pickle=True)
    ms_dataset = np.load(args.ms_labels, allow_pickle=True)
    pan_rows = np.asarray(pan_dataset[1])
    ms_rows = aligned_ms_rows(pan_rows, np.asarray(ms_dataset[1]))
    if pan_rows.ndim != 2 or pan_rows.shape[1] < 11 or ms_rows.shape[1] < 11:
        raise ValueError('Both inputs must contain the 11-column multi-sensor split table.')

    # Columns 6/7 are PAN prediction/confidence; columns 8/9 are MS.
    pan_prediction = pan_rows[:, 6].astype(int)
    pan_confidence = pan_rows[:, 7].astype(float)
    ms_prediction = ms_rows[:, 8].astype(int)
    ms_confidence = ms_rows[:, 9].astype(float)

    is_crevasse = np.isin(pan_prediction, crevasse_ids)
    is_water = np.isin(ms_prediction, water_ids)
    selected = (
        is_crevasse
        & is_water
        & (pan_confidence >= pan_threshold)
        & (ms_confidence >= ms_threshold)
    )

    # Retain the original sensor-specific labels only for the selected
    # water-in-crevasse patches.  Keeping the 11-column shape makes this file
    # compatible with the existing split-image viewers and NetCDF exporter.
    output_rows = pan_rows.copy()
    output_rows[:, 8:10] = ms_rows[:, 8:10]
    output_rows[~selected, 6] = -1  # PAN label
    output_rows[~selected, 7] = 0   # PAN confidence
    output_rows[~selected, 8] = -1  # MS label
    output_rows[~selected, 9] = 0   # MS confidence

    output_dataset = np.empty(2, dtype=object)
    output_dataset[0] = pan_dataset[0]
    output_dataset[1] = output_rows
    if args.output is None:
        pan_stem, _ = os.path.splitext(args.pan_labels)
        args.output = f'{pan_stem}_water_crevasse_mask.npy'
    output_parent = os.path.dirname(args.output)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    np.save(args.output, output_dataset)

    print(f'PAN crevasse classes: {crevasse_names} (ids {crevasse_ids.tolist()})')
    print(f'MS water classes: {water_names} (ids {water_ids.tolist()})')
    print(f'Confidence thresholds: PAN >= {pan_threshold:.2f}, MS >= {ms_threshold:.2f}')
    print(
        f'Selected {int(selected.sum())} of {selected.size} patches as '
        f'{combined_label_name}; all other PAN/MS labels were masked to -1.'
    )
    print(f'Saved combined mask: {args.output}')


if __name__ == '__main__':
    main()
