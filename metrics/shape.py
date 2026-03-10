"""Shape metrics for buildings (compactness, elongation, etc.)."""

import geopandas as gpd
import momepy
import numpy as np
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

    Adjacent buildings are first dissolved into unified structures (same as
    courtyard_area_metrics) so that courtyards between touching buildings
    are reflected. Only structures that have interior holes (courtyards) are
    included; if none do, all courtyard_index statistics are set to NaN.
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
    if dissolved.geom_type == "MultiPolygon":
        structures = list(dissolved.geoms)
    else:
        structures = [dissolved]

    # Only consider structures that have interior holes (courtyards)
    structures_with_holes = [g for g in structures if len(g.interiors) > 0]
    if not structures_with_holes:
        p = "courtyard_index_"
        nan_stats = {f"{p}mean": np.nan, f"{p}std": np.nan}
        for q in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            nan_stats[f"{p}p{q}"] = np.nan
        for k, v in nan_stats.items():
            prepared.cell_gdf[k] = v
        return prepared.cell_gdf

    structures_gdf = gpd.GeoDataFrame(
        geometry=structures_with_holes, crs=buildings.crs
    )
    s = momepy.courtyard_index(structures_gdf)
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
    """Compute shape index = sqrt(area/pi) / (0.5 * longest axis). Effectively measures how close the shape is to a circle.

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

    # fails on multipolygons (https://github.com/pysal/momepy/issues/739)
    s = momepy.corners(prepared.conformal.explode())
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

    # fails on multipolygons (https://github.com/pysal/momepy/issues/739)
    s = momepy.squareness(prepared.conformal.explode())
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
    
    In building footprint analysis:

    High ERI → rectangular buildings (row houses, slab blocks, industrial sheds)

    Medium ERI → slightly articulated buildings

    Low ERI → complex footprints (L-shapes, courtyards, fragmented forms)
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
    
    For building footprints:

    High elongation (low ratio) → row houses, slabs, industrial strips

    Low elongation (near 1) → compact blocks, detached houses
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
    """Compute facade ratio (area / perimeter) for raw buildings and for unioned structures.

    Returns two sets of metrics:
    - facade_ratio_*: from raw building geometries.
    - facade_ratio_structure_*: from dissolved (union-all then explode) structures,
      so adjacent buildings are treated as one, matching courtyard metrics.

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant

    # Raw building metrics
    s_raw = momepy.facade_ratio(buildings)
    stats_raw = aggregate_stats(s_raw, prefix="facade_ratio")
    for k, v in stats_raw.items():
        prepared.cell_gdf[k] = v

    # Union and explode into structures (same as courtyard metrics)
    dissolved = buildings.union_all()
    if dissolved.geom_type == "MultiPolygon":
        structures = list(dissolved.geoms)
    else:
        structures = [dissolved]
    structures_gdf = gpd.GeoDataFrame(geometry=structures, crs=buildings.crs)
    s_struct = momepy.facade_ratio(structures_gdf)
    stats_struct = aggregate_stats(s_struct, prefix="facade_ratio_structure")
    for k, v in stats_struct.items():
        prepared.cell_gdf[k] = v

    return prepared.cell_gdf


def fractal_dimension_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute fractal dimension = 2*log(perimeter/4) / log(area) for raw buildings and unioned structures.

    Returns two sets of metrics:
    - fractal_dimension_*: from raw building geometries.
    - fractal_dimension_structure_*: from dissolved (union-all then explode) structures,
      so adjacent buildings are treated as one, matching courtyard metrics.

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.

    ~1.00	Very simple shape (square, rectangle)
    1.05–1.15	Moderately articulated
    >1.20	Complex or fragmented footprint
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant

    # Raw building metrics
    s_raw = momepy.fractal_dimension(buildings)
    stats_raw = aggregate_stats(s_raw, prefix="fractal_dimension")
    for k, v in stats_raw.items():
        prepared.cell_gdf[k] = v

    # Union and explode into structures (same as courtyard metrics)
    dissolved = buildings.union_all()
    if dissolved.geom_type == "MultiPolygon":
        structures = list(dissolved.geoms)
    else:
        structures = [dissolved]
    structures_gdf = gpd.GeoDataFrame(geometry=structures, crs=buildings.crs)
    s_struct = momepy.fractal_dimension(structures_gdf)
    stats_struct = aggregate_stats(s_struct, prefix="fractal_dimension_structure")
    for k, v in stats_struct.items():
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
    """Compute compactness-weighted axis (longest axis length + compactness) for raw buildings and unioned structures.

    Returns two sets of metrics:
    - compactness_weighted_axis_*: from raw building geometries.
    - compactness_weighted_axis_structure_*: from dissolved (union-all then explode) structures,
      so adjacent buildings are treated as one, matching courtyard metrics.

    Uses equidistant CRS for accurate distance computation.
    Measures how efficiently a polygon fills space relative to its principal axes.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant

    # Raw building metrics
    s_raw = momepy.compactness_weighted_axis(buildings)
    stats_raw = aggregate_stats(s_raw, prefix="compactness_weighted_axis")
    for k, v in stats_raw.items():
        prepared.cell_gdf[k] = v

    # Union and explode into structures (same as courtyard metrics)
    dissolved = buildings.union_all()
    if dissolved.geom_type == "MultiPolygon":
        structures = list(dissolved.geoms)
    else:
        structures = [dissolved]
    structures_gdf = gpd.GeoDataFrame(geometry=structures, crs=buildings.crs)
    s_struct = momepy.compactness_weighted_axis(structures_gdf)
    stats_struct = aggregate_stats(s_struct, prefix="compactness_weighted_axis_structure")
    for k, v in stats_struct.items():
        prepared.cell_gdf[k] = v

    return prepared.cell_gdf


def centroid_corner_distance_metrics(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute centroid-corner distance (mean and std of distances) for raw buildings and unioned structures.

    Returns two sets of metrics:
    - centroid_corner_distance_*: from raw building geometries.
    - centroid_corner_distance_structure_*: from dissolved (union-all then explode) structures,
      so adjacent buildings are treated as one, matching courtyard metrics.

    Uses equidistant CRS for accurate distance computation.
    """
    prepared = prepare_buildings(
        buildings_gdf, cell_polygon, equal_area_crs, equidistant_crs, conformal_crs
    )
    if prepared.equidistant.empty:
        return prepared.cell_gdf

    buildings = prepared.equidistant

    # Raw building metrics
    # fails on multipolygons (https://github.com/pysal/momepy/issues/739)
    df_raw = momepy.centroid_corner_distance(buildings.explode())
    for col in ["mean", "std"]:
        stats = aggregate_stats(df_raw[col], prefix=f"centroid_corner_distance_{col}")
        for k, v in stats.items():
            prepared.cell_gdf[k] = v

    # Union and explode into structures (same as courtyard metrics)
    dissolved = buildings.union_all()
    if dissolved.geom_type == "MultiPolygon":
        structures = list(dissolved.geoms)
    else:
        structures = [dissolved]
    structures_gdf = gpd.GeoDataFrame(geometry=structures, crs=buildings.crs)
    
    # fails on multipolygons (https://github.com/pysal/momepy/issues/739)
    df_struct = momepy.centroid_corner_distance(structures_gdf.explode())
    for col in ["mean", "std"]:
        stats = aggregate_stats(
            df_struct[col], prefix=f"centroid_corner_distance_structure_{col}"
        )
        for k, v in stats.items():
            prepared.cell_gdf[k] = v

    return prepared.cell_gdf
