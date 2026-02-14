"""Extract OSM data for polygons and compute urban morphology metrics."""

import hashlib
import logging
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
import quackosm as qosm
import requests
from pyproj import CRS
from shapely.geometry import Polygon

from metrics import compute_all_metrics
from metrics.compute import ALL_METRIC_NAMES
from metrics.street_networks import split_highways

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/preprocessed")


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


def _parse_selected_metrics(metrics: list[str] | str | None) -> set[str] | None:
    """Validate metric names, warning about unknowns."""
    if metrics is None:
        return None
    if isinstance(metrics, str):
        metrics = [metrics]
    selected = set(metrics)
    unknown = selected - set(ALL_METRIC_NAMES)
    if unknown:
        print(f"Warning: unknown metrics will be skipped: {', '.join(sorted(unknown))}")
        selected -= unknown
    return selected


def _get_region_ids(polygons_gdf: gpd.GeoDataFrame) -> list[str]:
    """Extract region_id values from the GeoDataFrame (index or column)."""
    if polygons_gdf.index.name == "region_id":
        return list(polygons_gdf.index)
    if "region_id" in polygons_gdf.columns:
        return list(polygons_gdf["region_id"])
    raise ValueError(
        "polygons_gdf must have a 'region_id' column or index. "
        f"Found columns: {list(polygons_gdf.columns)}, index name: {polygons_gdf.index.name}"
    )


def _cache_key(region_ids: list[str]) -> str:
    """Generate an MD5 hash from sorted region_id values."""
    joined = ",".join(sorted(str(rid) for rid in region_ids))
    return hashlib.md5(joined.encode()).hexdigest()


def _parse_height(val) -> float | None:
    """Parse an OSM height value (e.g. '15', '15m', '50 ft') into meters."""
    import re

    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    val = val.strip().lower()
    match = re.match(r"^([\d.]+)\s*(m|metres?|meters?|ft|feet)?", val)
    if match:
        num = float(match.group(1))
        unit = (match.group(2) or "m").lower()
        if "ft" in unit or "feet" in unit:
            num *= 0.3048
        return num
    return None


def _parse_levels(val) -> float | None:
    """Parse a building:levels value into a numeric count."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _add_building_columns(buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract height and building:levels from OSM tags into proper columns.

    Adds ``height`` (meters) and ``building_levels`` (count) columns.
    Values are parsed from the ``tags`` dict column or from exploded tag columns.
    """
    buildings = buildings_gdf.copy()
    has_tags = "tags" in buildings.columns

    # Parse height
    if "height" in buildings.columns:
        buildings["height"] = buildings["height"].apply(_parse_height)
    elif has_tags:
        buildings["height"] = buildings["tags"].apply(
            lambda t: _parse_height(t.get("height")) if isinstance(t, dict) else None
        )
    else:
        buildings["height"] = None

    # Parse building:levels
    if "building:levels" in buildings.columns:
        buildings["building_levels"] = buildings["building:levels"].apply(_parse_levels)
    elif has_tags:
        buildings["building_levels"] = buildings["tags"].apply(
            lambda t: _parse_levels(t.get("building:levels")) if isinstance(t, dict) else None
        )
    else:
        buildings["building_levels"] = None

    return buildings


def _extract_osm_data(pbf_file: str, geometry_filter):
    """Extract buildings, highways, and landuse from PBF for a geometry filter."""
    buildings_gdf = qosm.convert_pbf_to_geodataframe(
        pbf_file,
        tags_filter={"building": True},
        geometry_filter=geometry_filter,
        keep_all_tags=True,
    ).to_crs(4326)
    buildings_gdf = _add_building_columns(buildings_gdf)
    highways_gdf = qosm.convert_pbf_to_geodataframe(
        pbf_file,
        tags_filter={"highway": True},
        geometry_filter=geometry_filter,
        keep_all_tags=True,
    ).to_crs(4326)
    landuse_gdf = qosm.convert_pbf_to_geodataframe(
        pbf_file, tags_filter={"landuse": True}, geometry_filter=geometry_filter
    ).to_crs(4326)
    return buildings_gdf, highways_gdf, landuse_gdf


def _tag_with_regions(
    gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    region_ids: list[str],
) -> gpd.GeoDataFrame:
    """Add boolean columns indicating which region each feature intersects.

    For each region_id, adds a column ``region_id`` (the actual id string)
    with True/False values.
    """
    sorted_ids = sorted(region_ids)

    if gdf.empty:
        empty_cols = {rid: pd.Series(dtype=bool) for rid in sorted_ids}
        return pd.concat([gdf, pd.DataFrame(empty_cols)], axis=1)

    # Defragment the input to avoid PerformanceWarning
    gdf = gdf.copy()

    # Ensure polygons_gdf has region_id as a column for the spatial join
    polys = polygons_gdf.copy()
    if polys.index.name == "region_id":
        polys = polys.reset_index()

    # Spatial join to find which features intersect which polygons
    joined = gpd.sjoin(
        gdf[["geometry"]],
        polys[["region_id", "geometry"]],
        how="left",
        predicate="intersects",
    )

    # Build boolean columns in alphabetical order via concat
    bool_cols = {}
    for rid in sorted_ids:
        matching_indices = joined.index[joined["region_id"] == rid].unique()
        bool_cols[rid] = gdf.index.isin(matching_indices)

    return pd.concat([gdf, pd.DataFrame(bool_cols, index=gdf.index)], axis=1)


def _preprocess_osm_data(
    polygons_gdf: gpd.GeoDataFrame,
    pbf_file: str,
    region_ids: list[str],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Extract OSM data for the union of all polygons, tag with region membership, and cache.

    Returns:
        Tuple of (buildings_gdf, highways_gdf, vehicles_gdf, pedestrians_gdf, landuse_gdf),
        each with boolean columns per region_id.
    """
    cache_hash = _cache_key(region_ids)
    cache_path = CACHE_DIR / cache_hash
    file_names = ["buildings", "highways", "vehicles", "pedestrians", "landuse"]

    # Check cache
    if cache_path.exists() and all((cache_path / f"{n}.parquet").exists() for n in file_names):
        print(f"Loading preprocessed data from cache ({cache_hash[:8]}...)")
        buildings_gdf = gpd.read_parquet(cache_path / "buildings.parquet")
        highways_gdf = gpd.read_parquet(cache_path / "highways.parquet")
        vehicles_gdf = gpd.read_parquet(cache_path / "vehicles.parquet")
        pedestrians_gdf = gpd.read_parquet(cache_path / "pedestrians.parquet")
        landuse_gdf = gpd.read_parquet(cache_path / "landuse.parquet")
        return buildings_gdf, highways_gdf, vehicles_gdf, pedestrians_gdf, landuse_gdf

    # Extract OSM data using the union of all polygons
    print("Extracting OSM data for the combined area...")
    union_geom = polygons_gdf.union_all()
    buildings_gdf, highways_gdf, landuse_gdf = _extract_osm_data(pbf_file, union_geom)

    # Split highway networks
    print("Splitting highway networks...")
    vehicles_gdf, pedestrians_gdf = split_highways(highways_gdf)

    # Tag each dataset with boolean region membership
    print("Tagging features with region membership...")
    buildings_gdf = _tag_with_regions(buildings_gdf, polygons_gdf, region_ids)
    highways_gdf = _tag_with_regions(highways_gdf, polygons_gdf, region_ids)
    vehicles_gdf = _tag_with_regions(vehicles_gdf, polygons_gdf, region_ids)
    pedestrians_gdf = _tag_with_regions(pedestrians_gdf, polygons_gdf, region_ids)
    landuse_gdf = _tag_with_regions(landuse_gdf, polygons_gdf, region_ids)

    # Cache to disk
    cache_path.mkdir(parents=True, exist_ok=True)
    buildings_gdf.to_parquet(cache_path / "buildings.parquet", index=False)
    highways_gdf.to_parquet(cache_path / "highways.parquet", index=False)
    vehicles_gdf.to_parquet(cache_path / "vehicles.parquet", index=False)
    pedestrians_gdf.to_parquet(cache_path / "pedestrians.parquet", index=False)
    landuse_gdf.to_parquet(cache_path / "landuse.parquet", index=False)
    # Save the hexagon polygons with their region_ids
    hexagons = polygons_gdf[["geometry"]].copy()
    if polygons_gdf.index.name == "region_id":
        hexagons["region_id"] = polygons_gdf.index
    else:
        hexagons["region_id"] = polygons_gdf["region_id"].values
    hexagons.to_parquet(cache_path / "hexagons.parquet", index=False)
    print(f"Cached preprocessed data ({cache_hash[:8]}...)")

    return buildings_gdf, highways_gdf, vehicles_gdf, pedestrians_gdf, landuse_gdf


def _select_for_region(gdf: gpd.GeoDataFrame, region_id: str) -> gpd.GeoDataFrame:
    """Select features that intersect a given region and drop all region boolean columns."""
    if gdf.empty or region_id not in gdf.columns:
        return gdf
    selected = gdf[gdf[region_id]].copy()
    # Drop all region_id boolean columns to get a clean GeoDataFrame
    region_cols = [c for c in selected.columns if c in gdf.columns and gdf[c].dtype == "bool"]
    return selected.drop(columns=region_cols)


def compute_urban_morphometrics(
    polygons_gdf: gpd.GeoDataFrame,
    pbf_url: str,
    save_data_path: str | None = None,
    metrics: list[str] | str | None = None,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Extract OSM data and compute building morphology metrics for each polygon.

    OSM data is extracted once for the union of all polygons and tagged with
    boolean columns indicating which region each feature belongs to. This
    preprocessed data is cached on disk using an MD5 hash of the sorted
    region_id values, so subsequent runs with the same polygons skip extraction.

    Args:
        polygons_gdf: GeoDataFrame of polygons to compute metrics for.
            Must have a ``region_id`` column or index. Each row's geometry
            defines an analysis area.
        pbf_url: URL to the PBF file. Its basename is used to look for a cached copy.
        save_data_path: If provided, save buildings, networks, landuse, and the
            output metrics table to parquet files in this directory.
        metrics: List of metric names to compute (e.g.
            ["courtyard_area", "elongation", "degree_vehicle"]). A single
            string is auto-wrapped into a list. If omitted, all metrics
            are computed.
        equal_area_crs: CRS for area calculations (e.g. EPSG code, CRS object).
            Defaults to estimated UTM.
        equidistant_crs: CRS for distance/length calculations. Defaults to
            estimated UTM.
        conformal_crs: CRS for angular calculations. Defaults to estimated UTM.

    Returns:
        GeoDataFrame with one row per input polygon and all metric columns.
    """
    pbf_file = ensure_pbf(pbf_url)
    selected_metrics = _parse_selected_metrics(metrics)

    # Ensure CRS is WGS84
    if polygons_gdf.crs is not None and polygons_gdf.crs != "EPSG:4326":
        polygons_gdf = polygons_gdf.to_crs(4326)

    region_ids = _get_region_ids(polygons_gdf)

    # Preprocess: extract, tag, and cache OSM data
    buildings_all, highways_all, vehicles_all, pedestrians_all, landuse_all = (
        _preprocess_osm_data(polygons_gdf, pbf_file, region_ids)
    )

    # Optionally save the raw data
    if save_data_path is not None:
        out_dir = Path(save_data_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        buildings_all.to_parquet(out_dir / "buildings.parquet", index=False)
        vehicles_all.to_parquet(out_dir / "vehicles.parquet", index=False)
        pedestrians_all.to_parquet(out_dir / "pedestrians.parquet", index=False)
        landuse_all.to_parquet(out_dir / "landuse.parquet", index=False)
        print(f"Saved buildings, networks, and landuse to {out_dir}")

    # Resolve polygon geometries alongside their region_ids
    if polygons_gdf.index.name == "region_id":
        polys_iter = list(zip(polygons_gdf.index, polygons_gdf.geometry))
    else:
        polys_iter = list(zip(polygons_gdf["region_id"], polygons_gdf.geometry))

    # Log once which metrics will be computed
    if selected_metrics is not None:
        logger.info("Computing metrics: %s", ", ".join(sorted(selected_metrics)))
    else:
        logger.info("Computing all metrics")

    # Compute metrics for each polygon using pre-filtered data
    total = len(polys_iter)
    rows = []
    for i, (region_id, polygon) in enumerate(polys_iter):
        print(f"Computing metrics for {region_id} ({i + 1}/{total})...")

        buildings_gdf = _select_for_region(buildings_all, region_id)
        highways_gdf = _select_for_region(highways_all, region_id)
        vehicles_gdf = _select_for_region(vehicles_all, region_id)
        pedestrians_gdf = _select_for_region(pedestrians_all, region_id)

        result = compute_all_metrics(
            buildings_gdf,
            highways_gdf,
            vehicles_gdf,
            pedestrians_gdf,
            polygon,
            return_dict=False,
            selected_metrics=selected_metrics,
            quiet=True,
            equal_area_crs=equal_area_crs,
            equidistant_crs=equidistant_crs,
            conformal_crs=conformal_crs,
        )

        row = {}
        for col in result.columns:
            if col != "geometry":
                row[col] = result[col].iloc[0]
        rows.append(row)

    result = gpd.GeoDataFrame(
        pd.DataFrame(rows),
        geometry=polygons_gdf.geometry.values,
        crs=polygons_gdf.crs,
    )
    # Preserve region_id in the output
    if polygons_gdf.index.name == "region_id":
        result.index = polygons_gdf.index
    else:
        result["region_id"] = region_ids

    print(f"Computed {len(result.columns) - 1} metric columns for {total} polygons")

    if save_data_path is not None:
        out_dir = Path(save_data_path)
        result.to_parquet(out_dir / "output.parquet", index=True)
        print(f"Saved output to {out_dir / 'output.parquet'}")

    return result
