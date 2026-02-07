"""Dimension metrics for buildings (area, volume, perimeter, etc.)."""

import geopandas as gpd
import momepy
from shapely.geometry import Polygon

from ._utils import aggregate_stats, prepare_buildings


def courtyard_area_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute courtyard area metrics for buildings in the cell.

    Courtyard area is the area of interior holes (courtyards) within building
    polygons. Buildings with atria or interior courtyards have non-zero values.
    Captures the degree to which buildings incorporate open interior space.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of courtyard area per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.courtyard_area(buildings)
    stats = aggregate_stats(s, prefix="courtyard_area", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def floor_area_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute floor area metrics for buildings in the cell.

    Floor area estimates total usable floor space as area × (height // floor_height).
    Default floor height is 3m. Reflects building capacity and intensity of use.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of floor area per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.floor_area(buildings["area"], buildings["height"])
    stats = aggregate_stats(s, prefix="floor_area", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def longest_axis_length_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute longest axis length metrics for buildings in the cell.

    The longest axis is the diameter of the minimum bounding circle. Indicates
    the dominant scale of buildings and their linear extent.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of longest axis length per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.longest_axis_length(buildings)
    stats = aggregate_stats(s, prefix="longest_axis_length", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def perimeter_wall_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute perimeter wall length metrics for buildings in the cell.

    For standalone buildings, this is the exterior perimeter. For buildings
    that share walls (contiguous structures), it measures the perimeter of
    the joined structure divided among constituent buildings. Reflects
    building contiguity and street-facing facade length.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of perimeter wall length per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.perimeter_wall(buildings)
    stats = aggregate_stats(s, prefix="perimeter_wall", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def volume_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute volume metrics for buildings in the cell.

    Volume = area × height. Represents built mass and urban density.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of volume per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.volume(buildings["area"], buildings["height"])
    stats = aggregate_stats(s, prefix="volume", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf
