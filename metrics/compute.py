"""Aggregate all metrics into a single GeoDataFrame."""

import logging
import time

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

from .dimension import (
    courtyard_area_metrics,
    floor_area_metrics,
    longest_axis_length_metrics,
    perimeter_wall_metrics,
    volume_metrics,
)
from .distribution import (
    alignment_metrics,
    building_adjacency_metrics,
    cell_alignment_metrics,
    mean_interbuilding_distance_metrics,
    neighbor_distance_metrics,
    neighbors_metrics,
    orientation_metrics,
    shared_walls_metrics,
    street_alignment_metrics,
)
from .intensity import courtyards_metrics
from .shape import (
    centroid_corner_distance_metrics,
    circular_compactness_metrics,
    compactness_weighted_axis_metrics,
    convexity_metrics,
    corners_metrics,
    courtyard_index_metrics,
    elongation_metrics,
    equivalent_rectangular_index_metrics,
    facade_ratio_metrics,
    form_factor_metrics,
    fractal_dimension_metrics,
    rectangularity_metrics,
    shape_index_metrics,
    square_compactness_metrics,
    squareness_metrics,
)
from .street_relationship import nearest_street_distance_metrics, street_profile_metrics
from .street_connectivity import (
    GRAPH_METRICS,
    MODES,
    NODE_METRICS,
    PROPORTION_KEYS,
    compute_connectivity_metrics_by_name,
)

# Registry of building/street metrics: (name, function, arg_style)
# arg_style: "buildings" = (buildings_gdf, cell_polygon, ea, ed, cf)
#             "streets"  = (buildings_gdf, highways_gdf, cell_polygon, ea, ed, cf)
_BUILDING_METRICS = [
    ("courtyard_area", courtyard_area_metrics, "buildings"),
    ("floor_area", floor_area_metrics, "buildings"),
    ("longest_axis_length", longest_axis_length_metrics, "buildings"),
    ("perimeter_wall", perimeter_wall_metrics, "buildings"),
    ("volume", volume_metrics, "buildings"),
    ("circular_compactness", circular_compactness_metrics, "buildings"),
    ("square_compactness", square_compactness_metrics, "buildings"),
    ("convexity", convexity_metrics, "buildings"),
    ("courtyard_index", courtyard_index_metrics, "buildings"),
    ("rectangularity", rectangularity_metrics, "buildings"),
    ("shape_index", shape_index_metrics, "buildings"),
    ("corners", corners_metrics, "buildings"),
    ("squareness", squareness_metrics, "buildings"),
    ("equivalent_rectangular_index", equivalent_rectangular_index_metrics, "buildings"),
    ("elongation", elongation_metrics, "buildings"),
    ("facade_ratio", facade_ratio_metrics, "buildings"),
    ("fractal_dimension", fractal_dimension_metrics, "buildings"),
    ("form_factor", form_factor_metrics, "buildings"),
    ("compactness_weighted_axis", compactness_weighted_axis_metrics, "buildings"),
    ("centroid_corner_distance", centroid_corner_distance_metrics, "buildings"),
    ("orientation", orientation_metrics, "buildings"),
    ("shared_walls", shared_walls_metrics, "buildings"),
    ("alignment", alignment_metrics, "buildings"),
    ("neighbor_distance", neighbor_distance_metrics, "buildings"),
    ("mean_interbuilding_distance", mean_interbuilding_distance_metrics, "buildings"),
    ("building_adjacency", building_adjacency_metrics, "buildings"),
    ("neighbors", neighbors_metrics, "buildings"),
    ("cell_alignment", cell_alignment_metrics, "buildings"),
    ("street_alignment", street_alignment_metrics, "streets"),
    ("courtyards", courtyards_metrics, "buildings"),
    ("street_profile", street_profile_metrics, "streets"),
    ("nearest_street_distance", nearest_street_distance_metrics, "streets"),
]

# All possible connectivity metric names (for matching selected_metrics)
_CONNECTIVITY_METRIC_NAMES: list[str] = []
for _m in NODE_METRICS:
    for _mode in MODES:
        _CONNECTIVITY_METRIC_NAMES.append(f"{_m}_{_mode}")
for _m in GRAPH_METRICS:
    for _mode in MODES:
        _CONNECTIVITY_METRIC_NAMES.append(f"{_m}_{_mode}")
for _k in PROPORTION_KEYS:
    for _mode in MODES:
        _CONNECTIVITY_METRIC_NAMES.append(f"proportion_{_k}_{_mode}")

ALL_METRIC_NAMES: list[str] = [name for name, _, _ in _BUILDING_METRICS] + _CONNECTIVITY_METRIC_NAMES


def _configure_logging(
    level: int = logging.INFO,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure logging for the metrics module if not already configured."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=format)


def _run_metric(
    cell_gdf: gpd.GeoDataFrame | None,
    fn,
    name: str,
    *args,
    merge: bool = True,
    quiet: bool = False,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Run a metric function and optionally merge results, with start/finish logging."""
    if not quiet:
        logger.info("Computing metric: %s", name)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if merge and cell_gdf is not None and not result.empty:
            _merge_columns(cell_gdf, result)
        if not quiet:
            logger.info("Finished metric: %s (%.2fs)", name, elapsed)
        return result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning("Metric %s failed after %.2fs: %s", name, elapsed, e)
        raise


def compute_all_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    highways_gdf: gpd.GeoDataFrame,
    vehicles_gdf: gpd.GeoDataFrame,
    pedestrians_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    return_dict: bool = False,
    selected_metrics: set[str] | None = None,
    quiet: bool = False,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame]:
    """Compute building morphology and street connectivity metrics.

    Args:
        buildings_gdf: Buildings GeoDataFrame from OSM.
        highways_gdf: Full highways GeoDataFrame (for street alignment, profile, etc.).
        vehicles_gdf: Pre-filtered vehicle network.
        pedestrians_gdf: Pre-filtered pedestrian network.
        cell_polygon: The H3 cell polygon for the analysis area.
        return_dict: If True, return a dict mapping metric names to individual
            GeoDataFrames (one row each). If False (default), return a single
            GeoDataFrame with all metrics merged.
        selected_metrics: If provided, only compute metrics whose names are in this set.
            If None, compute all metrics.
        quiet: If True, suppress per-metric log messages.
        equal_area_crs: CRS for area calculations. Defaults to estimated UTM.
        equidistant_crs: CRS for distance calculations. Defaults to estimated UTM.
        conformal_crs: CRS for angular calculations. Defaults to estimated UTM.

    Returns:
        If return_dict=False: GeoDataFrame with one row, geometry=cell_polygon,
            and all metric columns. If return_dict=True: dict[str, GeoDataFrame]
            with keys like "courtyard_area", "floor_area", etc.
    """
    _configure_logging()

    crs_kwargs = dict(
        equal_area_crs=equal_area_crs,
        equidistant_crs=equidistant_crs,
        conformal_crs=conformal_crs,
    )

    merge = not return_dict
    metrics_dict: dict[str, gpd.GeoDataFrame] = {}
    cell_gdf: gpd.GeoDataFrame | None = None

    for name, fn, arg_style in _BUILDING_METRICS:
        if selected_metrics is not None and name not in selected_metrics:
            continue

        if arg_style == "streets":
            args = (buildings_gdf, highways_gdf, cell_polygon)
        else:
            args = (buildings_gdf, cell_polygon)
        # Append CRS kwargs to positional args
        args = (*args, equal_area_crs, equidistant_crs, conformal_crs)

        if cell_gdf is None:
            # First metric establishes the base GeoDataFrame
            if not quiet:
                logger.info("Computing metric: %s", name)
            t0 = time.perf_counter()
            cell_gdf = fn(*args)
            if not quiet:
                logger.info("Finished metric: %s (%.2fs)", name, time.perf_counter() - t0)
            if return_dict:
                metrics_dict[name] = cell_gdf.copy()
        else:
            result = _run_metric(cell_gdf, fn, name, *args, merge=merge, quiet=quiet)
            if return_dict:
                metrics_dict[name] = result

    # Ensure we have a base cell_gdf even if all building metrics were skipped
    if cell_gdf is None:
        from ._utils import prepare_buildings
        prepared = prepare_buildings(
            buildings_gdf, cell_polygon,
            equal_area_crs, equidistant_crs, conformal_crs,
        )
        cell_gdf = prepared.cell_gdf

    # Street connectivity metrics
    any_connectivity = selected_metrics is None or any(
        name in selected_metrics for name in _CONNECTIVITY_METRIC_NAMES
    )
    if any_connectivity:
        if not quiet:
            logger.info("Computing metric: connectivity")
        t0 = time.perf_counter()
        connectivity_results = compute_connectivity_metrics_by_name(
            vehicles_gdf, pedestrians_gdf, cell_polygon,
            equidistant_crs=equidistant_crs, conformal_crs=conformal_crs,
        )
        if not quiet:
            logger.info("Finished metric: connectivity (%.2fs)", time.perf_counter() - t0)
        for metric_name, result in connectivity_results.items():
            if selected_metrics is not None and metric_name not in selected_metrics:
                continue
            if merge and cell_gdf is not None and not result.empty:
                _merge_columns(cell_gdf, result)
            if return_dict:
                metrics_dict[metric_name] = result

    return metrics_dict if return_dict else cell_gdf


def _merge_columns(target: gpd.GeoDataFrame, source: gpd.GeoDataFrame) -> None:
    """Merge non-geometry columns from source into target."""
    for col in source.columns:
        if col != "geometry" and col not in target.columns:
            target[col] = source[col].iloc[0]
