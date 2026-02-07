"""Intensity metrics for buildings."""

import geopandas as gpd
import momepy
from libpysal.graph import Graph
from shapely.geometry import Polygon

from ._utils import aggregate_stats, prepare_buildings


def courtyards_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute courtyards count metrics for buildings in the cell.

    For each building (or joined contiguous structure), counts the number of
    interior rings (courtyards) in the union of buffered building geometries.
    Buildings that share walls are merged; the result is the courtyard count
    of the joined structure.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of courtyard count per building/structure.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    try:
        contiguity = Graph.build_contiguity(buildings, rook=True)
    except Exception:
        return cell_gdf

    s = momepy.courtyards(buildings, contiguity)
    stats = aggregate_stats(s, prefix="courtyards", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf
