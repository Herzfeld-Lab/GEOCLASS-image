# Jakobshavn WorldView Labeling

This directory is currently organized around one active labeling workflow for the
Jakobshavn ice stream WorldView subset.

## Active Files

- Active config: `jakobshavn-ice-stream-2018-2021-6class-labeling.config`
- Active split dataset: `jakobshavn-ice-stream-2018-2021-6class-no-multigenerational-subset_3648_(900,1200)_split.npy`
- Active training image folder: `Training_data/Greenland/Jakobshavn-wv-2018-2021-6class-no-multigenerational-from-npy`
- Active training indices: `jakobshavn-ice-stream-2018-2021-6class-labeling_6_314train_indices.npy`
- Active contour: `jak-ice-stream.npy`

Older configs and generated split datasets have been archived under
`backups/archived-2026-06-20/`. They were not deleted.

## Source Images

The active dataset contains 3,648 split tiles from four selected WorldView images:

| Image ID | Date | Tiles | Source image |
|---:|---|---:|---|
| 0 | 2018-07-30 | 1708 | `WV03_20180730150838_104001003FCB3C00_18JUL30150838-P1BS-502531210010_01_P008_u16rf3413.tif` |
| 1 | 2019-06-08 | 709 | `WV02_20190608234438_10300100931B9100_19JUN08234438-P1BS-503348257090_01_P002_u16rf3413.tif` |
| 2 | 2020-06-28 | 646 | `WV02_20200628000458_10300100A9364500_20JUN28000458-P1BS-504470829070_01_P001_u16rf3413.tif` |
| 3 | 2021-06-02 | 585 | `WV02_20210602233443_10300100BF726D00_21JUN02233443-P1BS-505459986070_01_P002_u16rf3413.tif` |

The 2022 candidate image was skipped because the current contour/split rules
produced no valid split tiles for it.

## Classes

The active config uses six classes. The previous `Multigenerational` class was
removed because there were only 9 examples; those tiles are now treated as
unlabeled in the active `.npy`.

| ID | Class |
|---:|---|
| 0 | Smooth surface, shear deformation |
| 1 | Closed conjugate |
| 2 | Closed compressional, shear |
| 3 | Open extensional, shear |
| 4 | Chaos |
| 5 | Wavy crevasses |

## Current Labeling Status

As of 2026-06-20, the active `.npy` dataset contains 393 labeled tiles and
3,255 unlabeled tiles:

| Class ID | Labeled tiles |
|---:|---:|
| 0 | 130 |
| 1 | 77 |
| 2 | 59 |
| 3 | 60 |
| 4 | 32 |
| 5 | 35 |

The active train/validation split is stratified: 314 training chips and 79
validation chips. Validation keeps at least 7 examples of every class.

## Commands

Run the labeling GUI:

```bash
python Split_Image_Explorer.py Config/greenland_enthalpy/jakobshavn-wv/jakobshavn-ice-stream-2018-2021-6class-labeling.config
```

Train the current VarioMLP config:

```bash
python train.py Config/greenland_enthalpy/jakobshavn-wv/jakobshavn-ice-stream-2018-2021-6class-labeling.config
```

Generate predictions from a new 6-class checkpoint:

```bash
python test.py Config/greenland_enthalpy/jakobshavn-wv/jakobshavn-ice-stream-2018-2021-6class-labeling.config \
  --load_checkpoint Output/<new_6class_run>/checkpoints/<epoch> \
  --output_dir Output/<new_6class_run>
```

## Current Training Notes

The first VarioMLP pilot was trained in `Output/greenland_enthalpy_19-06-2026_20:58`.
The best validation checkpoint was `checkpoints/epoch_30`.

That model performed poorly for model-assisted labeling: predictions were heavily
biased toward classes 0 and 3, did not predict several crevasse classes, and all
prediction confidences were below 50%. Use this run for debugging the workflow,
not as a trustworthy classifier. Its checkpoint has 7 output classes and should
not be loaded with the current 6-class config.

The old VarioMLP warning to use `(201, 268)` chips is stale for the current
configuration. With `vario_num_lag: 100`, `900x1200` and `450x600` chips produce
the 300-feature vectors expected by VarioMLP. A true `201x268` chip produces
159 features with `silas_directional_vario`, so it requires either
`vario_num_lag: 53` or a different variogram path.
