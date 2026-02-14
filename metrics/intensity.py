"""Intensity metrics for buildings."""

import geopandas as gpd
import momepy
from libpysal.graph import Graph
from pyproj import CRS
from shapely.geometry import Polygon

from ._utils import aggregate_stats, prepare_buildings


def courtyards_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute courtyards count metrics (number of interior rings).

    Primarily topological. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant
    try:
        contiguity = Graph.build_contiguity(buildings, rook=True)
    except Exception:
        return prepared.cell_gdf

    s = momepy.courtyards(buildings, contiguity)
    stats = aggregate_stats(s, prefix="courtyards", include_sum=True)
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf
