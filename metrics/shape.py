"""Shape metrics for buildings (compactness, elongation, etc.)."""

import geopandas as gpd
import momepy
from pyproj import CRS
from shapely.geometry import Polygon

from ._utils import aggregate_stats, prepare_buildings


def circular_compactness_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute circular compactness (area / area of enclosing circle).

    Uses equal-area CRS for accurate area computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equal_area.empty:
        return prepared.cell_gdf

    s = momepy.circular_compactness(prepared.equal_area)
    stats = aggregate_stats(s, prefix="circular_compactness")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def square_compactness_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute square compactness = (4*sqrt(area) / perimeter)^2.

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.square_compactness(prepared.equidistant)
    stats = aggregate_stats(s, prefix="square_compactness")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def convexity_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute convexity (area / area of convex hull).

    Uses equal-area CRS for accurate area computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equal_area.empty:
        return prepared.cell_gdf

    s = momepy.convexity(prepared.equal_area)
    stats = aggregate_stats(s, prefix="convexity")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def courtyard_index_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute courtyard index (courtyard area / total area).

    Uses equal-area CRS for accurate area computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equal_area.empty:
        return prepared.cell_gdf

    s = momepy.courtyard_index(prepared.equal_area)
    stats = aggregate_stats(s, prefix="courtyard_index")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def rectangularity_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute rectangularity (area / area of minimum rotated rectangle).

    Uses equal-area CRS for accurate area computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equal_area.empty:
        return prepared.cell_gdf

    s = momepy.rectangularity(prepared.equal_area)
    stats = aggregate_stats(s, prefix="rectangularity")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def shape_index_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute shape index = sqrt(area/pi) / (0.5 * longest axis).

    Dimensionless ratio mixing area and length. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.shape_index(prepared.equidistant)
    stats = aggregate_stats(s, prefix="shape_index")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def corners_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute corners count (vertices where angle deviates from 180 deg).

    Uses conformal CRS for accurate angle computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.conformal.empty:
        return prepared.cell_gdf

    s = momepy.corners(prepared.conformal)
    stats = aggregate_stats(s, prefix="corners", include_sum=True)
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def squareness_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute squareness (mean deviation of corner angles from 90 deg).

    Uses conformal CRS for accurate angle computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.conformal.empty:
        return prepared.cell_gdf

    s = momepy.squareness(prepared.conformal)
    stats = aggregate_stats(s, prefix="squareness")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def equivalent_rectangular_index_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute equivalent rectangular index (area and perimeter ratios).

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.equivalent_rectangular_index(prepared.equidistant)
    stats = aggregate_stats(s, prefix="equivalent_rectangular_index")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def elongation_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute elongation (shorter/longer side of minimum bounding rectangle).

    Dimensionless ratio. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.elongation(prepared.equidistant)
    stats = aggregate_stats(s, prefix="elongation")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def facade_ratio_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute facade ratio (area / perimeter).

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.facade_ratio(prepared.equidistant)
    stats = aggregate_stats(s, prefix="facade_ratio")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def fractal_dimension_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute fractal dimension = 2*log(perimeter/4) / log(area).

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.fractal_dimension(prepared.equidistant)
    stats = aggregate_stats(s, prefix="fractal_dimension")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def form_factor_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute form factor = surface / volume^(2/3).

    Uses equal-area CRS for accurate surface area and volume computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equal_area.empty:
        return prepared.cell_gdf

    buildings = prepared.equal_area
    s = momepy.form_factor(buildings, buildings["height"])
    stats = aggregate_stats(s, prefix="form_factor")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def compactness_weighted_axis_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute compactness-weighted axis (longest axis length + compactness).

    Uses equidistant CRS for accurate distance computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    s = momepy.compactness_weighted_axis(prepared.equidistant)
    stats = aggregate_stats(s, prefix="compactness_weighted_axis")
    for k, v in stats.items():
        prepared.cell_gdf[k] = v
    return prepared.cell_gdf


def centroid_corner_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute centroid-corner distance (mean and std of distances).

    Uses equidistant CRS for accurate distance computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    df = momepy.centroid_corner_distance(prepared.equidistant)
    for col in ["mean", "std"]:
        stats = aggregate_stats(df[col], prefix=f"centroid_corner_distance_{col}")
        for k, v in stats.items():
            prepared.cell_gdf[k] = v
    return prepared.cell_gdf
