#!/usr/bin/env python3
"""Extract OSM data for an H3 cell and display on a map."""

from pathlib import Path
from urllib.parse import urlparse

import fire
import h3
import quackosm as qosm
import requests
from shapely.geometry import Polygon

from metrics import compute_all_metrics
from metrics.street_networks import split_highways


def ensure_pbf(pbf_url: str) -> str:
    """Use cached PBF if it exists (by URL basename), otherwise download it.

    Args:
        pbf_url: URL to the PBF file. The basename is used to look for a cached copy.

    Returns:
        The path to the PBF file.
    """
    cache_path = Path(urlparse(pbf_url).path).name
    path = Path(cache_path)
    if path.exists():
        print(f"Using cached PBF: {path}")
        return str(path.resolve())

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PBF from {pbf_url}...")
    response = requests.get(pbf_url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {path}")
    return str(path.resolve())


def compute_urban_morphometrics(
    lat: float,
    lon: float,
    pbf_url: str,
    h3_level: int = 8,
    return_dict: bool = False,
    save_data_path: str | None = None,
):
    """Extract OSM data for an H3 cell and compute building morphology metrics.

    Args:
        lat: Latitude of the center point.
        lon: Longitude of the center point.
        pbf_url: URL to the PBF file. Its basename is used to look for a cached copy first.
        h3_level: H3 resolution level (0-15).
        return_dict: If True, return a dict mapping metric names to individual GeoDataFrames.
            If False (default), return a single GeoDataFrame with all metrics merged.
        save_data_path: If provided, save buildings, networks (vehicles, pedestrians),
            and landuse to geoparquet files in this directory.

    Returns:
        GeoDataFrame with one row (cell_polygon) and all metric columns, or dict of
        metric name -> GeoDataFrame when return_dict=True.
    """
    pbf_file = ensure_pbf(pbf_url)

    cell = h3.latlng_to_cell(lat, lon, h3_level)
    cell_boundary = h3.cell_to_boundary(cell)
    cell_polygon = Polygon(cell_boundary)

    print("Converting PBF to geodataframes...")
    buildings_gdf = qosm.convert_pbf_to_geodataframe(
        pbf_file, tags_filter={"building": True}, geometry_filter=cell_polygon
    )
    highways_gdf = qosm.convert_pbf_to_geodataframe(
        pbf_file,
        tags_filter={"highway": True},
        geometry_filter=cell_polygon,
        keep_all_tags=True,
    )
    landuse_gdf = qosm.convert_pbf_to_geodataframe(
        pbf_file, tags_filter={"landuse": True}, geometry_filter=cell_polygon
    )

    vehicles_gdf, pedestrians_gdf = split_highways(highways_gdf)

    if save_data_path is not None:
        out_dir = Path(save_data_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        buildings_gdf.to_parquet(out_dir / "buildings.parquet", index=False)
        vehicles_gdf.to_parquet(out_dir / "vehicles.parquet", index=False)
        pedestrians_gdf.to_parquet(out_dir / "pedestrians.parquet", index=False)
        landuse_gdf.to_parquet(out_dir / "landuse.parquet", index=False)
        print(f"Saved buildings, networks, and landuse to {out_dir}")

    result = compute_all_metrics(
        buildings_gdf,
        highways_gdf,
        vehicles_gdf,
        pedestrians_gdf,
        cell_polygon,
        return_dict=return_dict,
    )
    if return_dict:
        print(f"Computed {len(result)} metrics")
    else:
        print(f"Computed {len(result.columns) - 1} metric columns")
    return result


if __name__ == "__main__":
    fire.Fire(compute_urban_morphometrics)
