"""Intensity metrics for buildings."""

import geopandas as gpd
import momepy
from libpysal.graph import Graph

from ._utils import CellContext, aggregate_stats


def courtyards_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute courtyards count metrics (number of interior rings).

    Neighbourhood-aware: uses neighbourhood buildings to correctly compute
    contiguity at cell boundaries, then filters aggregation to focal buildings.

    Primarily topological. Uses equidistant CRS.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.equidistant.index
    buildings = ctx.neighbourhood_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    try:
        contiguity = Graph.build_contiguity(buildings, rook=True)
    except Exception:
        return cell_gdf

    s = momepy.courtyards(buildings, contiguity)
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal, prefix="courtyards", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = ctx.focal_buildings.equidistant[["geometry"]].copy()
        _d["courtyards"] = s_focal
        ctx.dump("courtyards", _d)
    return cell_gdf
