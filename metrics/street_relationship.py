"""Metrics capturing the relationship between buildings and streets."""

import geopandas as gpd
import momepy
from pyproj import CRS
from shapely.geometry import Polygon

from ._utils import aggregate_stats, prepare_buildings, prepare_highways


def street_profile_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    highways_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
    distance: float = 10,
    tick_length: float = 50,
) -> gpd.GeoDataFrame:
    """Compute street profile metrics (width, openness, height-width ratio).

    Uses equidistant CRS for accurate width/distance computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    highways_prep = prepare_highways(
        highways_gdf, cell_polygon, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty or highways_prep.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant
    highways = highways_prep.equidistant

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
            prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def nearest_street_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    highways_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
    max_distance: float = 200,
) -> gpd.GeoDataFrame:
    """Compute distance to nearest street.

    Uses equidistant CRS for accurate distance computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    highways_prep = prepare_highways(
        highways_gdf, cell_polygon, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty or highways_prep.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant
    highways = highways_prep.equidistant

    dist_series = buildings.geometry.apply(
        lambda g: highways.geometry.distance(g).min()
    )
    stats = aggregate_stats(dist_series.dropna(), prefix="nearest_street_distance")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf
