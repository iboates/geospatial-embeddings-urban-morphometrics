"""Aggregate all metrics into a single GeoDataFrame."""

import logging
import time

import geopandas as gpd
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
from .street_connectivity import compute_connectivity_metrics_by_name


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
    **kwargs,
) -> gpd.GeoDataFrame:
    """Run a metric function and optionally merge results, with start/finish logging."""
    logger.info("Computing metric: %s", name)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if merge and cell_gdf is not None and not result.empty:
            _merge_columns(cell_gdf, result)
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
) -> gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame]:
    """Compute all building morphology metrics.

    Args:
        buildings_gdf: Buildings GeoDataFrame from OSM.
        highways_gdf: Full highways GeoDataFrame (for street alignment, profile, etc.).
        vehicles_gdf: Pre-filtered vehicle network.
        pedestrians_gdf: Pre-filtered pedestrian network.
        cell_polygon: The H3 cell polygon for the analysis area.
        return_dict: If True, return a dict mapping metric names to individual
            GeoDataFrames (one row each). If False (default), return a single
            GeoDataFrame with all metrics merged.

    Returns:
        If return_dict=False: GeoDataFrame with one row, geometry=cell_polygon,
            and all metric columns. If return_dict=True: dict[str, GeoDataFrame]
            with keys like "courtyard_area", "floor_area", etc.
    """
    _configure_logging()

    merge = not return_dict
    metrics_dict: dict[str, gpd.GeoDataFrame] = {}

    # Dimension metrics (first establishes base with geometry in projected CRS)
    logger.info("Computing metric: courtyard_area")
    t0 = time.perf_counter()
    cell_gdf = courtyard_area_metrics(buildings_gdf, cell_polygon)
    logger.info("Finished metric: courtyard_area (%.2fs)", time.perf_counter() - t0)
    if return_dict:
        metrics_dict["courtyard_area"] = cell_gdf.copy()

    for fn, name in [
        (floor_area_metrics, "floor_area"),
        (longest_axis_length_metrics, "longest_axis_length"),
        (perimeter_wall_metrics, "perimeter_wall"),
        (volume_metrics, "volume"),
    ]:
        result = _run_metric(cell_gdf, fn, name, buildings_gdf, cell_polygon, merge=merge)
        if return_dict:
            metrics_dict[name] = result

    # Shape metrics
    for fn, name in [
        (circular_compactness_metrics, "circular_compactness"),
        (square_compactness_metrics, "square_compactness"),
        (convexity_metrics, "convexity"),
        (courtyard_index_metrics, "courtyard_index"),
        (rectangularity_metrics, "rectangularity"),
        (shape_index_metrics, "shape_index"),
        (corners_metrics, "corners"),
        (squareness_metrics, "squareness"),
        (equivalent_rectangular_index_metrics, "equivalent_rectangular_index"),
        (elongation_metrics, "elongation"),
        (facade_ratio_metrics, "facade_ratio"),
        (fractal_dimension_metrics, "fractal_dimension"),
        (form_factor_metrics, "form_factor"),
        (compactness_weighted_axis_metrics, "compactness_weighted_axis"),
        (centroid_corner_distance_metrics, "centroid_corner_distance"),
    ]:
        result = _run_metric(cell_gdf, fn, name, buildings_gdf, cell_polygon, merge=merge)
        if return_dict:
            metrics_dict[name] = result

    # Distribution metrics
    for fn, name in [
        (orientation_metrics, "orientation"),
        (shared_walls_metrics, "shared_walls"),
        (alignment_metrics, "alignment"),
        (neighbor_distance_metrics, "neighbor_distance"),
        (mean_interbuilding_distance_metrics, "mean_interbuilding_distance"),
        (building_adjacency_metrics, "building_adjacency"),
        (neighbors_metrics, "neighbors"),
        (cell_alignment_metrics, "cell_alignment"),
    ]:
        result = _run_metric(cell_gdf, fn, name, buildings_gdf, cell_polygon, merge=merge)
        if return_dict:
            metrics_dict[name] = result

    result = _run_metric(
        cell_gdf,
        street_alignment_metrics,
        "street_alignment",
        buildings_gdf,
        highways_gdf,
        cell_polygon,
        merge=merge,
    )
    if return_dict:
        metrics_dict["street_alignment"] = result

    # Intensity metrics
    result = _run_metric(
        cell_gdf, courtyards_metrics, "courtyards", buildings_gdf, cell_polygon, merge=merge
    )
    if return_dict:
        metrics_dict["courtyards"] = result

    # Building-street relationship metrics
    result = _run_metric(
        cell_gdf,
        street_profile_metrics,
        "street_profile",
        buildings_gdf,
        highways_gdf,
        cell_polygon,
        merge=merge,
    )
    if return_dict:
        metrics_dict["street_profile"] = result
    result = _run_metric(
        cell_gdf,
        nearest_street_distance_metrics,
        "nearest_street_distance",
        buildings_gdf,
        highways_gdf,
        cell_polygon,
        merge=merge,
    )
    if return_dict:
        metrics_dict["nearest_street_distance"] = result

    # Street connectivity metrics (degree, meshedness, gamma, etc. - same level as other metrics)
    logger.info("Computing metric: connectivity")
    t0 = time.perf_counter()
    connectivity_results = compute_connectivity_metrics_by_name(
        vehicles_gdf, pedestrians_gdf, cell_polygon
    )
    logger.info("Finished metric: connectivity (%.2fs)", time.perf_counter() - t0)
    for metric_name, result in connectivity_results.items():
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
