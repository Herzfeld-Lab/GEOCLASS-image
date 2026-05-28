import rasterio
import numpy as np
from pathlib import Path


def find_ms_tiff(folder):
    """
    Find the multispectral TIFF (M1BS) in a folder.
    """
    for f in folder.glob("*.tif"):
        name = f.name.upper()
        if "M1BS" in name:  # multispectral indicator
            return f
    return None


def compute_wri(input_tif, output_tif):
    """
    Compute WRI = (band5 + band3) / (band7 + band8)
    and save as float32 GeoTIFF.
    """

    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        nodata = src.nodata

        # Output will be single band float32
        profile.update({
            "count": 1,
            "dtype": "float32",
            "compress": "lzw"
        })

        with rasterio.open(output_tif, "w", **profile) as dst:

            # Process in chunks
            for _, window in src.block_windows(1):

                data = src.read(window=window).astype(np.float32)

                # WV band indexing (0-based)
                # band 3 = index 2 (Green)
                # band 5 = index 4 (Red)
                # band 7 = index 6 (NIR1)
                # band 8 = index 7 (NIR2)

                green = data[2]
                red   = data[4]
                nir   = data[6]
                mir   = data[7]

                numerator = green + red
                denominator = nir + mir

                # Avoid divide-by-zero
                wri = np.zeros_like(numerator, dtype=np.float32)
                valid = denominator != 0

                wri[valid] = numerator[valid] / denominator[valid]

                # Handle nodata
                if nodata is not None:
                    nodata_mask = (
                        (data[2] == nodata) |
                        (data[4] == nodata) |
                        (data[6] == nodata) |
                        (data[7] == nodata)
                    )
                    wri[nodata_mask] = np.nan

                dst.write(wri, 1, window=window)


def process_parent_folder(parent_dir, output_dir):
    parent_dir = Path(parent_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {parent_dir}")

    for subfolder in parent_dir.iterdir():
        if not subfolder.is_dir():
            continue

        print(f"\nProcessing folder: {subfolder.name}")

        ms_tif = find_ms_tiff(subfolder)

        if ms_tif is None:
            print("  No multispectral TIFF found, skipping.")
            continue

        output_path = output_dir / f"{subfolder.name}_WRI.tif"

        try:
            compute_wri(ms_tif, output_path)
            print(f"  WRI saved: {output_path.name}")
        except Exception as e:
            print(f"  ERROR: {e}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    process_parent_folder(
        parent_dir=r"C:\Users\Silas\Desktop\ws\NN_Class\Data\negri_dataset\test",
        output_dir=r"C:\Users\Silas\Desktop\ws\NN_Class\Data\negri_dataset\WRI_outputs"
    )