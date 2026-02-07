"""Spatial distribution metrics for buildings."""

import geopandas as gpd
import momepy
from libpysal.graph import Graph
from shapely.geometry import Polygon

from ._utils import aggregate_stats, prepare_buildings, prepare_highways


def _build_graphs(buildings: gpd.GeoDataFrame):
    """Build tessellation, contiguity, and neighborhood graphs for distribution metrics."""
    try:
        limit = momepy.buffered_limit(buildings)
        tessellation = momepy.morphological_tessellation(buildings, clip=limit)
    except Exception:
        return None, None, None

    try:
        contiguity = Graph.build_contiguity(tessellation, rook=False)
    except Exception:
        contiguity = None

    try:
        knn = Graph.build_knn(buildings.centroid, k=min(15, len(buildings)))
    except Exception:
        knn = None

    return tessellation, contiguity, knn


def orientation_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute building orientation metrics in the cell.

    Orientation is the deviation of the minimum bounding rectangle from
    cardinal directions (0–45°). Indicates alignment to street grid.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of orientation per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.orientation(buildings)
    stats = aggregate_stats(s, prefix="orientation")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def shared_walls_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute shared wall length metrics for buildings in the cell.

    Measures the length of walls shared with adjacent buildings. Indicates
    degree of building contiguity and party-wall construction.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of shared wall length per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.shared_walls(buildings)
    stats = aggregate_stats(s, prefix="shared_walls", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def alignment_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute alignment metrics (orientation consistency among neighbors).

    Mean deviation of building orientation from adjacent buildings in a
    Delaunay triangulation. Lower values indicate more uniform alignment.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of alignment per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    tessellation, contiguity, knn = _build_graphs(buildings)
    if knn is None:
        return cell_gdf

    orientation = momepy.orientation(buildings)
    knn_aligned = knn.assign_self_weight()
    s = momepy.alignment(orientation, knn_aligned)
    stats = aggregate_stats(s, prefix="alignment")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def neighbor_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute mean distance to adjacent buildings metrics.

    For each building, measures mean distance to neighbors in a Delaunay
    triangulation. Indicates building spacing and density.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of neighbor distance per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    try:
        tri = Graph.build_triangulation(buildings.centroid)
    except Exception:
        return cell_gdf

    s = momepy.neighbor_distance(buildings, tri)
    stats = aggregate_stats(s.dropna(), prefix="neighbor_distance")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def mean_interbuilding_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute mean interbuilding distance metrics.

    For each building, mean distance between adjacent buildings within its
    neighborhood. Requires adjacency (Delaunay) and neighborhood (KNN) graphs.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of mean interbuilding distance per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    try:
        tri = Graph.build_triangulation(buildings.centroid)
        knn = Graph.build_knn(buildings.centroid, k=min(15, len(buildings)))
    except Exception:
        return cell_gdf

    s = momepy.mean_interbuilding_distance(buildings, tri, knn)
    stats = aggregate_stats(s, prefix="mean_interbuilding_distance")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def building_adjacency_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute building adjacency metrics.

    Ratio of joined structures (contiguous buildings) to buildings in
    neighborhood. Higher values indicate more buildings share walls.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of building adjacency per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    tessellation, contiguity, knn = _build_graphs(buildings)
    if contiguity is None or knn is None:
        return cell_gdf

    # Build contiguity on buildings (rook) for adjacency
    try:
        building_contig = Graph.build_contiguity(buildings, rook=True)
    except Exception:
        return cell_gdf

    s = momepy.building_adjacency(building_contig, knn.assign_self_weight())
    stats = aggregate_stats(s, prefix="building_adjacency")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def neighbors_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute number of neighbors metrics (from tessellation contiguity).

    Count of adjacent tessellation cells per building. Indicates how many
    buildings border each one.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of neighbor count per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    tessellation, contiguity, _ = _build_graphs(buildings)
    if tessellation is None or contiguity is None:
        return cell_gdf

    s = momepy.neighbors(tessellation, contiguity)
    stats = aggregate_stats(s, prefix="neighbors", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def street_alignment_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    highways_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute street alignment metrics (building vs. street orientation).

    Deviation of building orientation from the nearest street's orientation.
    Lower values indicate buildings aligned with the street grid.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of street alignment per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    highways = prepare_highways(highways_gdf, cell_polygon)
    if buildings.empty or highways.empty:
        return cell_gdf

    street_idx = momepy.get_nearest_street(buildings, highways, max_distance=200)
    valid = street_idx.notna()
    if not valid.any():
        return cell_gdf

    blg_orient = momepy.orientation(buildings)
    str_orient = momepy.orientation(highways)
    s = momepy.street_alignment(blg_orient, str_orient, street_idx)
    stats = aggregate_stats(s.dropna(), prefix="street_alignment")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def cell_alignment_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute cell alignment metrics (building vs. tessellation cell orientation).

    Deviation of building orientation from its morphological tessellation cell.
    Indicates how well buildings follow the Voronoi-derived cell orientation.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of cell alignment per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    tessellation, contiguity, _ = _build_graphs(buildings)
    if tessellation is None:
        return cell_gdf

    blg_orient = momepy.orientation(buildings)
    tess_orient = momepy.orientation(tessellation)
    s = momepy.cell_alignment(tess_orient, blg_orient)
    stats = aggregate_stats(s, prefix="cell_alignment")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf
