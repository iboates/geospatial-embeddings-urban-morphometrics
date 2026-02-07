"""Shape metrics for buildings (compactness, elongation, etc.)."""

import geopandas as gpd
import momepy
from shapely.geometry import Polygon

from ._utils import aggregate_stats, prepare_buildings


def circular_compactness_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute circular compactness metrics for buildings in the cell.

    Circular compactness = area / area of enclosing circle. Values range 0–1;
    a circle has 1, elongated shapes have lower values. Measures how
    compact vs. spread out buildings are.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of circular compactness per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.circular_compactness(buildings)
    stats = aggregate_stats(s, prefix="circular_compactness")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def square_compactness_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute square compactness metrics for buildings in the cell.

    Square compactness = (4√area / perimeter)². Values range 0–1; a square
    has 1. Common in urban morphology for measuring regularity.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of square compactness per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.square_compactness(buildings)
    stats = aggregate_stats(s, prefix="square_compactness")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def convexity_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute convexity metrics for buildings in the cell.

    Convexity = area / area of convex hull. Values range 0–1; convex
    polygons have 1. Indicates concavity (e.g. L-shapes, courtyards).

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of convexity per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.convexity(buildings)
    stats = aggregate_stats(s, prefix="convexity")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def courtyard_index_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute courtyard index metrics for buildings in the cell.

    Courtyard index = courtyard area / total area. Proportion of building
    footprint devoted to interior courtyards. Higher values indicate
    atrium-style or courtyard buildings.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of courtyard index per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.courtyard_index(buildings)
    stats = aggregate_stats(s, prefix="courtyard_index")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def rectangularity_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute rectangularity metrics for buildings in the cell.

    Rectangularity = area / area of minimum rotated rectangle. Values 0–1;
    a rectangle has 1. Measures how close buildings are to rectangular form.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of rectangularity per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.rectangularity(buildings)
    stats = aggregate_stats(s, prefix="rectangularity")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def shape_index_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute shape index metrics for buildings in the cell.

    Shape index = √(area/π) / (0.5 × longest axis). Compares the radius of
    an equivalent circle to half the longest axis. Values near 1 indicate
    compact, round shapes.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of shape index per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.shape_index(buildings)
    stats = aggregate_stats(s, prefix="shape_index")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def corners_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute corners count metrics for buildings in the cell.

    Counts vertices where the angle deviates from 180° by more than 10°.
    Indicates complexity of building footprints.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of corner count per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.corners(buildings)
    stats = aggregate_stats(s, prefix="corners", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def squareness_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute squareness metrics for buildings in the cell.

    Squareness is the mean deviation of corner angles from 90°. Lower values
    indicate more rectangular (right-angled) corners.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of squareness per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.squareness(buildings)
    stats = aggregate_stats(s, prefix="squareness")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def equivalent_rectangular_index_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute equivalent rectangular index metrics for buildings in the cell.

    Combines area and perimeter ratios with the minimum bounding rectangle.
    Measures how well a shape approximates a rectangle in both area and form.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of equivalent rectangular index per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.equivalent_rectangular_index(buildings)
    stats = aggregate_stats(s, prefix="equivalent_rectangular_index")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def elongation_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute elongation metrics for buildings in the cell.

    Elongation is the ratio of the shorter to longer side of the minimum
    bounding rectangle. Values 0–1; squares have 1, lines approach 0.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of elongation per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.elongation(buildings)
    stats = aggregate_stats(s, prefix="elongation")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def facade_ratio_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute facade ratio metrics for buildings in the cell.

    Facade ratio = area / perimeter. Larger values indicate more compact
    footprints with less perimeter per unit area.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of facade ratio per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.facade_ratio(buildings)
    stats = aggregate_stats(s, prefix="facade_ratio")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def fractal_dimension_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute fractal dimension metrics for buildings in the cell.

    Fractal dimension = 2×log(perimeter/4) / log(area). Measures complexity
    of the boundary; values typically 1–2; higher values indicate more
    irregular outlines.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of fractal dimension per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.fractal_dimension(buildings)
    stats = aggregate_stats(s, prefix="fractal_dimension")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def form_factor_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute form factor metrics for buildings in the cell.

    Form factor = surface / volume^(2/3), where surface includes walls and roof.
    Lower values indicate more compact, cube-like buildings.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of form factor per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.form_factor(buildings, buildings["height"])
    stats = aggregate_stats(s, prefix="form_factor")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def compactness_weighted_axis_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute compactness-weighted axis metrics for buildings in the cell.

    Combines longest axis length with a compactness term. Captures both
    scale and shape in a single measure.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean, std,
    and deciles (p10–p90) of compactness-weighted axis per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    s = momepy.compactness_weighted_axis(buildings)
    stats = aggregate_stats(s, prefix="compactness_weighted_axis")
    for k, v in stats.items():
        cell_gdf[k] = v
    return cell_gdf


def centroid_corner_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Compute centroid-corner distance metrics for buildings in the cell.

    Measures mean and std of distances from centroid to corners. Indicates
    spread and uniformity of building footprint.

    Returns a GeoDataFrame with one row (cell_polygon) and columns: mean_mean,
    mean_std, std_mean, std_std, and deciles for centroid-corner distance
    mean and std per building.
    """
    buildings, cell_gdf = prepare_buildings(buildings_gdf, cell_polygon)
    if buildings.empty:
        return cell_gdf

    df = momepy.centroid_corner_distance(buildings)
    for col in ["mean", "std"]:
        stats = aggregate_stats(df[col], prefix=f"centroid_corner_distance_{col}")
        for k, v in stats.items():
            cell_gdf[k] = v
    return cell_gdf
