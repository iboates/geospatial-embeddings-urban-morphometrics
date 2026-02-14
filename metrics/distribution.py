"""Spatial distribution metrics for buildings."""

import geopandas as gpd
import momepy
from libpysal.graph import Graph
from pyproj import CRS
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
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute building orientation (deviation from cardinal directions).

    Uses conformal CRS for accurate angle computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.conformal.empty:
        return prepared.cell_gdf

    s = momepy.orientation(prepared.conformal)
    stats = aggregate_stats(s, prefix="orientation")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def shared_walls_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute shared wall length between adjacent buildings.

    Uses equidistant CRS for accurate length computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.shared_walls(prepared.equidistant)
    stats = aggregate_stats(s, prefix="shared_walls", include_sum=True)
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def alignment_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute alignment (orientation consistency among neighbors).

    Uses conformal CRS for accurate angle computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.conformal.empty:
        return prepared.cell_gdf

    buildings = prepared.conformal
    tessellation, contiguity, knn = _build_graphs(buildings)
    if knn is None:
        return prepared.cell_gdf

    orientation = momepy.orientation(buildings)
    knn_aligned = knn.assign_self_weight()
    s = momepy.alignment(orientation, knn_aligned)
    stats = aggregate_stats(s, prefix="alignment")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def neighbor_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute mean distance to adjacent buildings.

    Uses equidistant CRS for accurate distance computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant
    try:
        tri = Graph.build_triangulation(buildings.centroid)
    except Exception:
        return prepared.cell_gdf

    s = momepy.neighbor_distance(buildings, tri)
    stats = aggregate_stats(s.dropna(), prefix="neighbor_distance")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def mean_interbuilding_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute mean interbuilding distance.

    Uses equidistant CRS for accurate distance computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant
    try:
        tri = Graph.build_triangulation(buildings.centroid)
        knn = Graph.build_knn(buildings.centroid, k=min(15, len(buildings)))
    except Exception:
        return prepared.cell_gdf

    s = momepy.mean_interbuilding_distance(buildings, tri, knn)
    stats = aggregate_stats(s, prefix="mean_interbuilding_distance")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def building_adjacency_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute building adjacency (ratio of joined structures).

    Primarily topological. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant
    tessellation, contiguity, knn = _build_graphs(buildings)
    if contiguity is None or knn is None:
        return prepared.cell_gdf

    try:
        building_contig = Graph.build_contiguity(buildings, rook=True)
    except Exception:
        return prepared.cell_gdf

    s = momepy.building_adjacency(building_contig, knn.assign_self_weight())
    stats = aggregate_stats(s, prefix="building_adjacency")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def neighbors_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute number of tessellation neighbors.

    Primarily topological. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant
    tessellation, contiguity, _ = _build_graphs(buildings)
    if tessellation is None or contiguity is None:
        return prepared.cell_gdf

    s = momepy.neighbors(tessellation, contiguity)
    stats = aggregate_stats(s, prefix="neighbors", include_sum=True)
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def street_alignment_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    highways_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute street alignment (building vs. street orientation).

    Uses conformal CRS for accurate angle computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    highways_prep = prepare_highways(
        highways_gdf, cell_polygon, equidistant_crs, conformal_crs
    )
    if prepared.conformal.empty or highways_prep.conformal.empty:
        return prepared.cell_gdf

    buildings = prepared.conformal
    highways = highways_prep.conformal

    street_idx = momepy.get_nearest_street(buildings, highways, max_distance=200)
    valid = street_idx.notna()
    if not valid.any():
        return prepared.cell_gdf

    blg_orient = momepy.orientation(buildings)
    str_orient = momepy.orientation(highways)
    s = momepy.street_alignment(blg_orient, str_orient, street_idx)
    stats = aggregate_stats(s.dropna(), prefix="street_alignment")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def cell_alignment_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute cell alignment (building vs. tessellation cell orientation).

    Uses conformal CRS for accurate angle computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.conformal.empty:
        return prepared.cell_gdf

    buildings = prepared.conformal
    tessellation, contiguity, _ = _build_graphs(buildings)
    if tessellation is None:
        return prepared.cell_gdf

    blg_orient = momepy.orientation(buildings)
    tess_orient = momepy.orientation(tessellation)
    s = momepy.cell_alignment(tess_orient, blg_orient)
    stats = aggregate_stats(s, prefix="cell_alignment")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf
