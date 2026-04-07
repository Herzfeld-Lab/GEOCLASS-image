import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

# Input GeoJSON paths
avannarleq_json_file = "/Volumes/rachel_hdd/greenland-enthalpy/data/arcgis-online-map/avannarleq-glacier.geojson"
jakobshavn_json_file = "/Volumes/rachel_hdd/greenland-enthalpy/data/arcgis-online-map/jakobshavn-glacier.geojson"

HERE = Path(__file__).resolve().parent
avannarleq_npy_file = HERE / "avannarleq-s2" / "avannarleq_contour.npy"
jakobshavn_npy_file = HERE / "jakobshavn-s2" / "jakobshavn_contour.npy"

# check projection of avannarleq_json_file and jakobshavn_json_file
with open(avannarleq_json_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    print(data.get("crs"))
with open(jakobshavn_json_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    print(data.get("crs"))

# GeoJSON is WGS84 lon/lat; pipeline expects UTM coordinates (see utm_epsg_code in configs).
WGS84_TO_UTM22N = Transformer.from_crs("EPSG:3413", "EPSG:32622", always_xy=True)



def lonlat_to_utm22n(xy: np.ndarray) -> np.ndarray:
    """(N, 2) lon/lat -> (N, 2) easting/northing in WGS 84 / UTM zone 22N."""
    e, n = WGS84_TO_UTM22N.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([e, n])


def _exterior_ring_xy(geometry: dict) -> np.ndarray:
    """Exterior ring as (N, 2) float array. Handles Polygon and MultiPolygon."""
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    if gtype == "Polygon":
        ring = coords[0]
    elif gtype == "MultiPolygon":
        ring = coords[0][0]
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")
    arr = np.asarray(ring, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected exterior ring shape (N, 2), got {arr.shape}")
    return arr


def coords_from_geojson(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") == "FeatureCollection":
        geometry = data["features"][0]["geometry"]
    elif data.get("type") == "Feature":
        geometry = data["geometry"]
    else:
        geometry = data
    return _exterior_ring_xy(geometry)


def main() -> None:
    avannarleq_npy = lonlat_to_utm22n(coords_from_geojson(avannarleq_json_file))
    jakobshavn_npy = lonlat_to_utm22n(coords_from_geojson(jakobshavn_json_file))

    avannarleq_npy_file.parent.mkdir(parents=True, exist_ok=True)
    jakobshavn_npy_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(avannarleq_npy_file, avannarleq_npy)
    np.save(jakobshavn_npy_file, jakobshavn_npy)

    print(f"Wrote {avannarleq_npy_file} shape {avannarleq_npy.shape}")
    print(f"Wrote {jakobshavn_npy_file} shape {jakobshavn_npy.shape}")

    # print first 10 coordinates
    print(avannarleq_npy[:10])
    print(jakobshavn_npy[:10])


if __name__ == "__main__":
    main()
