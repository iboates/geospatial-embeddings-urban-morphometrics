"""Spatial distribution metrics for buildings."""

import geopandas as gpd
import momepy
from libpysal.graph import Graph

from ._utils import CellContext, aggregate_stats


def orientation_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute building orientation (deviation from cardinal directions).

    Uses conformal CRS for accurate angle computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.conformal
    if buildings.empty:
        return cell_gdf

    s = momepy.orientation(buildings)
    stats = aggregate_stats(s, prefix="orientation")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("orientation", buildings[["geometry"]].assign(orientation=s))
    return cell_gdf


def shared_walls_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute shared wall length between adjacent buildings.

    Returns two sets of metrics:
    - shared_walls_*: absolute shared wall length.
    - shared_walls_ratio_*: shared wall length as ratio of building perimeter.

    Neighbourhood-aware: uses neighbourhood buildings to correctly compute shared
    walls at cell boundaries, then filters aggregation to focal buildings.

    Uses equidistant CRS for accurate length computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.equidistant.index
    buildings = ctx.neighbourhood_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    # Absolute shared wall length
    s = momepy.shared_walls(buildings)
    s.name = "shared_walls"
    s_focal = s[s.index.isin(focal_idx)]

    stats = aggregate_stats(s_focal, prefix="shared_walls", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v

    # Shared wall length as ratio of building perimeter
    perimeter = buildings.geometry.length
    ratio = s / perimeter
    ratio_focal = ratio[ratio.index.isin(focal_idx)]
    stats_ratio = aggregate_stats(ratio_focal, prefix="shared_walls_ratio", include_sum=True)
    for k, v in stats_ratio.items():
        cell_gdf[k] = v

    if ctx.dump_dir is not None:
        _d = ctx.focal_buildings.equidistant[["geometry"]].copy()
        _d["shared_walls"] = s_focal
        _d["shared_walls_ratio"] = ratio_focal
        ctx.dump("shared_walls", _d)

    return cell_gdf


def alignment_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute alignment (orientation consistency among neighbors).

    Neighbourhood-aware: uses neighbourhood buildings and ctx.knn graph,
    then filters aggregation to focal buildings.

    Uses conformal CRS for accurate angle computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.conformal.index
    buildings = ctx.neighbourhood_buildings.conformal
    if buildings.empty:
        return cell_gdf

    knn = ctx.knn
    if knn is None:
        return cell_gdf

    orientation = momepy.orientation(buildings)
    knn_aligned = knn.assign_self_weight()
    s = momepy.alignment(orientation, knn_aligned)
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal, prefix="alignment")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = ctx.focal_buildings.conformal[["geometry"]].copy()
        _d["alignment"] = s_focal
        ctx.dump("alignment", _d)
    return cell_gdf


def neighbor_distance_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute mean distance to adjacent buildings.

    Neighbourhood-aware: builds triangulation from neighbourhood buildings,
    then filters aggregation to focal buildings.

    Uses equidistant CRS for accurate distance computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.equidistant.index
    buildings = ctx.neighbourhood_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    try:
        tri = Graph.build_triangulation(buildings.centroid)
    except Exception:
        return cell_gdf

    s = momepy.neighbor_distance(buildings, tri)
    s.name = "neighbor_distance"
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal.dropna(), prefix="neighbor_distance")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = ctx.focal_buildings.equidistant[["geometry"]].copy()
        _d["neighbor_distance"] = s_focal
        ctx.dump("neighbor_distance", _d)
    return cell_gdf


def mean_interbuilding_distance_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute mean interbuilding distance.

    Neighbourhood-aware: builds triangulation from neighbourhood buildings and
    uses ctx.knn, then filters aggregation to focal buildings.

    Uses equidistant CRS for accurate distance computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.equidistant.index
    buildings = ctx.neighbourhood_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    knn = ctx.knn
    if knn is None:
        return cell_gdf

    try:
        tri = Graph.build_triangulation(buildings.centroid)
    except Exception:
        return cell_gdf

    s = momepy.mean_interbuilding_distance(buildings, tri, knn)
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal, prefix="mean_interbuilding_distance")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = ctx.focal_buildings.equidistant[["geometry"]].copy()
        _d["mean_interbuilding_distance"] = s_focal
        ctx.dump("mean_interbuilding_distance", _d)
    return cell_gdf


def building_adjacency_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute building adjacency (ratio of joined structures).

    Neighbourhood-aware: uses neighbourhood buildings and ctx.knn,
    then filters aggregation to focal buildings.

    Primarily topological. Uses equidistant CRS.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.equidistant.index
    buildings = ctx.neighbourhood_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    knn = ctx.knn
    if knn is None:
        return cell_gdf

    try:
        building_contig = Graph.build_contiguity(buildings, rook=True)
    except Exception:
        return cell_gdf

    s = momepy.building_adjacency(building_contig, knn.assign_self_weight())
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal, prefix="building_adjacency")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = ctx.focal_buildings.equidistant[["geometry"]].copy()
        _d["building_adjacency"] = s_focal
        ctx.dump("building_adjacency", _d)
    return cell_gdf


def neighbors_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute number of tessellation neighbors.

    Neighbourhood-aware: uses ctx.tessellation and ctx.contiguity (both built
    from neighbourhood buildings), then filters aggregation to focal buildings.

    Primarily topological. Uses equidistant CRS.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.equidistant.index

    tessellation = ctx.tessellation
    contiguity = ctx.contiguity
    if tessellation is None or contiguity is None:
        return cell_gdf

    s = momepy.neighbors(tessellation, contiguity)
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal, prefix="neighbors", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        tess_focal = tessellation[tessellation.index.isin(focal_idx)]
        ctx.dump("neighbors", tess_focal[["geometry"]].assign(neighbors=s_focal))
    return cell_gdf


def street_alignment_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute street alignment (building vs. street orientation).

    Neighbourhood-aware: uses neighbourhood buildings and highways (conformal),
    then filters aggregation to focal buildings.

    Uses conformal CRS for accurate angle computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.conformal.index
    buildings = ctx.neighbourhood_buildings.conformal
    highways = ctx.neighbourhood_highways.conformal

    if buildings.empty or highways.empty:
        return cell_gdf

    street_idx = momepy.get_nearest_street(buildings, highways, max_distance=500)
    valid = street_idx.notna()
    if not valid.any():
        return cell_gdf

    blg_orient = momepy.orientation(buildings)
    str_orient = momepy.orientation(highways)
    s = momepy.street_alignment(blg_orient, str_orient, street_idx)
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal.dropna(), prefix="street_alignment")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = ctx.focal_buildings.conformal[["geometry"]].copy()
        _d["street_alignment"] = s_focal
        ctx.dump("street_alignment", _d)
    return cell_gdf


def cell_alignment_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute cell alignment (building vs. tessellation cell orientation).

    Neighbourhood-aware: uses ctx.tessellation (built from neighbourhood buildings)
    and neighbourhood buildings conformal orientation, then filters to focal buildings.

    Uses conformal CRS for accurate angle computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.conformal.index
    buildings = ctx.neighbourhood_buildings.conformal

    tessellation = ctx.tessellation
    if tessellation is None or buildings.empty:
        return cell_gdf

    blg_orient = momepy.orientation(buildings)
    tess_orient = momepy.orientation(tessellation)
    s = momepy.cell_alignment(tess_orient, blg_orient)
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal, prefix="cell_alignment")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        tess_focal = tessellation[tessellation.index.isin(focal_idx)]
        ctx.dump("cell_alignment", tess_focal[["geometry"]].assign(cell_alignment=s_focal))
    return cell_gdf
