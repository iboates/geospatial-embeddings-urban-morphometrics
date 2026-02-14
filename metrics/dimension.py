"""Dimension metrics for buildings (area, volume, perimeter, etc.)."""

import geopandas as gpd
import momepy
from pyproj import CRS
from shapely.geometry import Polygon
import pandas as pd

from ._utils import PreparedBuildings, aggregate_stats, prepare_buildings


def courtyard_area_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute courtyard area metrics for buildings in the cell.

    Adjacent buildings are first dissolved into unified structures so that
    courtyards formed between touching buildings are captured. Each interior
    hole (courtyard) is then extracted as an individual polygon, so a single
    structure with multiple courtyards contributes multiple entries.
    Statistics (mean, std, deciles, sum) are computed per individual courtyard.
    A count of courtyards is also included.

    Uses equal-area CRS for accurate area computation.
    """
    
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equal_area.empty:
        return prepared.cell_gdf

    buildings = prepared.equal_area

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
    prepared.cell_gdf["courtyard_area_count"] = courtyard_count

    if courtyard_count > 0:
        s = pd.Series(courtyard_areas)
        stats = aggregate_stats(s, prefix="courtyard_area", include_sum=True)
        for k, v in stats.items():
            prepared.cell_gdf[k] = v

    return prepared.cell_gdf


def floor_area_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute floor area metrics for buildings in the cell.

    Floor area estimates total usable floor space as area x (height // floor_height).
    Uses equal-area CRS for accurate area computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equal_area.empty:
        return prepared.cell_gdf

    buildings = prepared.equal_area
    s = momepy.floor_area(buildings["area"], buildings["height"])
    stats = aggregate_stats(s, prefix="floor_area", include_sum=True)
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def longest_axis_length_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute longest axis length metrics for buildings in the cell.

    The longest axis is the diameter of the minimum bounding circle.
    Uses equidistant CRS for accurate distance computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.longest_axis_length(prepared.equidistant)
    stats = aggregate_stats(s, prefix="longest_axis_length", include_sum=True)
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def perimeter_wall_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute perimeter wall length metrics for buildings in the cell.

    Uses equidistant CRS for accurate perimeter/length computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.perimeter_wall(prepared.equidistant)
    stats = aggregate_stats(s, prefix="perimeter_wall", include_sum=True)
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def volume_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute volume metrics for buildings in the cell.

    Volume = area x height. Uses equal-area CRS for accurate area computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equal_area.empty:
        return prepared.cell_gdf

    buildings = prepared.equal_area
    s = momepy.volume(buildings["area"], buildings["height"])
    stats = aggregate_stats(s, prefix="volume", include_sum=True)
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf
