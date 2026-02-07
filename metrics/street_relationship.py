"""Metrics capturing the relationship between buildings and streets."""

import geopandas as gpd
import momepy
from shapely.geometry import Polygon

from ._utils import aggregate_stats, prepare_buildings, prepare_highways


def street_profile_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    highways_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    distance: float = 10,
    tick_length: float = 50,
) -> gpd.GeoDataFrame:
    """Compute street profile metrics (width, openness, height-width ratio).

    Analyzes perpendicular ticks along streets to measure street width,
    openness (proportion of ticks without buildings), and optionally
    height/width ratio. Captures the enclosure defined by buildings along streets.

    Returns a GeoDataFrame with one row (cell_polygon) and columns for mean,
    std, and deciles of width, openness, width_deviation, and if height
    available: height, height_deviation, hw_ratio.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    highways = prepare_highways(highways_gdf, cell_polygon)
    if buildings.empty or highways.empty:
        return cell_gdf

    height = buildings["height"] if "height" in buildings.columns else None
    df = momepy.street_profile(
        highways, buildings, distance=distance, tick_length=tick_length, height=height
    )

    summable_cols = {"width", "height"}
    for col in df.columns:
        stats = aggregate_stats(
            df[col].dropna(),
            prefix=f"street_profile_{col}",
            include_sum=col in summable_cols,
        )
        for k, v in stats.items():
            cell_gdf[k] = v
    return cell_gdf


def nearest_street_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    highways_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    max_distance: float = 200,
) -> gpd.GeoDataFrame:
    """Compute distance to nearest street metrics for buildings.

    For each building, the distance to the nearest street. Indicates how
    far buildings are from the street network (frontage, setbacks).

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of distance to nearest street per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    highways = prepare_highways(highways_gdf, cell_polygon)
    if buildings.empty or highways.empty:
        return cell_gdf

    dist_series = buildings.geometry.apply(
        lambda g: highways.geometry.distance(g).min()
    )
    stats = aggregate_stats(dist_series.dropna(), prefix="nearest_street_distance")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf
