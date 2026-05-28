import rasterio
import numpy as np
from pathlib import Path


def compute_minmax(path):
    stats = []

    with rasterio.open(path) as src:
        nodata = src.nodata

        for b in range(1, src.count + 1):
            band = src.read(b)

            if nodata is not None:
                valid = band[band != nodata]
            else:
                valid = band

            if valid.size == 0:
                stats.append((0, 1))
                continue

            stats.append((float(valid.min()), float(valid.max())))

    return stats


def match_min_and_rescale(reference_path, target_path, output_path):

    print("Computing stats...")

    ref_stats = compute_minmax(reference_path)
    tgt_stats = compute_minmax(target_path)

    with rasterio.open(target_path) as src:
        profile = src.profile.copy()
        tgt_nodata = src.nodata

        with rasterio.open(reference_path) as ref:
            ref_nodata = ref.nodata

        # ✅ keep dtype but update nodata to match reference
        profile.update({
            "dtype": src.dtypes[0],
            "nodata": ref_nodata,
            "compress": "lzw"
        })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:

            for _, window in src.block_windows(1):

                data = src.read(window=window).astype(np.float32)

                for b in range(data.shape[0]):
                    band = data[b]

                    tmin, tmax = tgt_stats[b]
                    rmin, rmax = ref_stats[b]

                    if tmax - tmin < 1e-6:
                        band[:] = rmin
                        continue

                    # --- create nodata mask BEFORE modifying ---
                    if tgt_nodata is not None:
                        nodata_mask = (band == tgt_nodata)
                    else:
                        nodata_mask = np.zeros_like(band, dtype=bool)

                    # --- shift + rescale (NO CLIPPING) ---
                    band -= np.float32(tmin)

                    scale = np.float32((rmax - rmin) / (tmax - tmin))
                    band *= scale

                    band += np.float32(rmin)

                    # --- assign reference nodata ---
                    if ref_nodata is not None:
                        band[nodata_mask] = ref_nodata

                # convert back to original dtype
                data = data.astype(profile["dtype"])

                dst.write(data, window=window)

    print(f"\n Done: {output_path}")

match_min_and_rescale(
    reference_path="/Users/Silas/Desktop/ws/NN_Class/Data/negri_dataset/20200801/WV02_20200801124058_10300100A9358200_20AUG01124058-M1BS-504570336050_01_P004_u16ns3413.tif",
    target_path="/Users/Silas/Desktop/ws/NN_Class/Data/negri_dataset/20250715/WV03_20250715114408_10400100AACC3D00_25JUL15114408-M1BS-509755891020_01_P004_u16rf3413.tif",
    output_path="/Users/Silas/Desktop/ws/NN_Class/Data/negri_dataset/test2/20250715/WV03_20250715114408_10400100AACC3D00_25JUL15114408-M1BS-509755891020_01_P004_u16rf3413.tif"
)