"""Dimension metrics for buildings (area, volume, perimeter, etc.)."""

import geopandas as gpd
import momepy
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

from ._utils import CellContext, aggregate_stats


def courtyard_area_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute courtyard area metrics for buildings in the cell.

    Adjacent buildings are first dissolved into unified structures so that
    courtyards formed between touching buildings are captured. Each interior
    hole (courtyard) is then extracted as an individual polygon, so a single
    structure with multiple courtyards contributes multiple entries.
    Statistics (mean, std, deciles, sum) are computed per individual courtyard.
    A count of courtyards is also included.

    Uses equal-area CRS for accurate area computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equal_area
    if buildings.empty:
        return cell_gdf

    # Dissolve adjacent (touching/overlapping) buildings into unified structures
    dissolved = buildings.union_all()
    # Explode multi-part geometries into individual polygons
    if dissolved.geom_type == "MultiPolygon":
        structures = list(dissolved.geoms)
    else:
        structures = [dissolved]

    # Extract each interior hole as an individual courtyard polygon
    courtyard_areas = []
    for structure in structures:
        for interior in structure.interiors:
            courtyard_poly = Polygon(interior)
            courtyard_areas.append(courtyard_poly.area)

    courtyard_count = len(courtyard_areas)
    cell_gdf["courtyard_area_count"] = courtyard_count

    if courtyard_count > 0:
        s = pd.Series(courtyard_areas)
        stats = aggregate_stats(s, prefix="courtyard_area", include_sum=True)
        for k, v in stats.items():
            cell_gdf[k] = v

    if ctx.dump_dir is not None and courtyard_count > 0:
        courtyard_polys = [Polygon(i) for g in structures for i in g.interiors]
        ctx.dump("courtyard_area", gpd.GeoDataFrame(
            {"area": courtyard_areas}, geometry=courtyard_polys, crs=buildings.crs
        ))

    return cell_gdf


def floor_area_metrics(ctx: CellContext, floor_height: float = 3.0) -> gpd.GeoDataFrame:
    """Compute floor area metrics for buildings in the cell.

    Floor area = footprint area x number of levels.
    The number of levels is determined by:
    1. ``building_levels`` column (from OSM ``building:levels`` tag) when available.
    2. Otherwise, ``height / floor_height`` (rounded down, minimum 1).

    Args:
        floor_height: Assumed height per floor in meters (default 3.0).

    Uses equal-area CRS for accurate area computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equal_area
    if buildings.empty:
        return cell_gdf

    # Determine number of levels per building
    if "building_levels" in buildings.columns:
        levels = buildings["building_levels"].copy()
    else:
        levels = pd.Series(np.nan, index=buildings.index)

    # Where levels are missing, derive from height / floor_height
    missing = levels.isna()
    if missing.any():
        derived = np.floor(buildings.loc[missing, "height"] / floor_height).clip(lower=1)
        levels.loc[missing] = derived

    s = buildings["area"] * levels
    stats = aggregate_stats(s, prefix="floor_area", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = buildings[["geometry"]].copy()
        _d["floor_area"] = s
        ctx.dump("floor_area", _d)
    return cell_gdf


def longest_axis_length_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute longest axis length metrics for buildings in the cell.

    The longest axis is the diameter of the minimum bounding circle.
    Uses equidistant CRS for accurate distance computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    s = momepy.longest_axis_length(buildings)
    stats = aggregate_stats(s, prefix="longest_axis_length", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = buildings[["geometry"]].copy()
        _d["longest_axis_length"] = s
        ctx.dump("longest_axis_length", _d)
    return cell_gdf


def perimeter_wall_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute perimeter wall length metrics for buildings in the cell. Returns the perimeter of the JOINED STRUCTURES, i.e. all buildings which are touching

    Neighbourhood-aware: uses neighbourhood buildings to correctly compute shared
    walls at cell boundaries, then filters aggregation to focal buildings.

    Uses equidistant CRS for accurate perimeter/length computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    focal_idx = ctx.focal_buildings.equidistant.index
    buildings = ctx.neighbourhood_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    s = momepy.perimeter_wall(buildings)
    s_focal = s[s.index.isin(focal_idx)]
    stats = aggregate_stats(s_focal, prefix="perimeter_wall", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = ctx.focal_buildings.equidistant[["geometry"]].copy()
        _d["perimeter_wall"] = s_focal
        ctx.dump("perimeter_wall", _d)
    return cell_gdf


def volume_metrics(ctx: CellContext, floor_height: float = 3.0) -> gpd.GeoDataFrame:
    """Compute volume metrics for buildings in the cell.

    Volume = footprint area x building height.
    The building height is determined by:
    1. ``height`` column (from OSM ``height`` tag) when available.
    2. ``building_levels * floor_height`` when levels are available but height is not.
    3. Default height (6m) otherwise.

    Args:
        floor_height: Assumed height per floor in meters (default 3.0).

    Uses equal-area CRS for accurate area computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equal_area
    if buildings.empty:
        return cell_gdf

    # Resolve height: prefer parsed height, then levels * floor_height, then default
    height = buildings["height"].copy()
    if "building_levels" in buildings.columns:
        from_levels = buildings["building_levels"] * floor_height
        missing = height.isna()
        height.loc[missing] = from_levels.loc[missing]
    height = height.fillna(6.0)

    s = buildings["area"] * height
    stats = aggregate_stats(s, prefix="volume", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        _d = buildings[["geometry"]].copy()
        _d["volume"] = s
        ctx.dump("volume", _d)
    return cell_gdf
