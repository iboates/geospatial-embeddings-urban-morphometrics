"""Shape metrics for buildings (compactness, elongation, etc.)."""

import geopandas as gpd
import momepy
import numpy as np

from ._utils import CellContext, aggregate_stats


def circular_compactness_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute circular compactness (area / area of enclosing circle).

    Uses equal-area CRS for accurate area computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equal_area
    if buildings.empty:
        return cell_gdf

    s = momepy.circular_compactness(buildings)
    stats = aggregate_stats(s, prefix="circular_compactness")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("circular_compactness", buildings[["geometry"]].assign(circular_compactness=s))
    return cell_gdf


def square_compactness_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute square compactness = (4*sqrt(area) / perimeter)^2.

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    s = momepy.square_compactness(buildings)
    stats = aggregate_stats(s, prefix="square_compactness")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("square_compactness", buildings[["geometry"]].assign(square_compactness=s))
    return cell_gdf


def convexity_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute convexity (area / area of convex hull).

    Uses equal-area CRS for accurate area computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equal_area
    if buildings.empty:
        return cell_gdf

    s = momepy.convexity(buildings)
    stats = aggregate_stats(s, prefix="convexity")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("convexity", buildings[["geometry"]].assign(convexity=s))
    return cell_gdf


def courtyard_index_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute courtyard index (courtyard area / total area).

    Adjacent buildings are first dissolved into unified structures (same as
    courtyard_area_metrics) so that courtyards between touching buildings
    are reflected. Only structures that have interior holes (courtyards) are
    included; if none do, all courtyard_index statistics are set to NaN.
    Uses equal-area CRS for accurate area computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equal_area
    if buildings.empty:
        return cell_gdf

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
            cell_gdf[k] = v
        return cell_gdf

    structures_gdf = gpd.GeoDataFrame(
        geometry=structures_with_holes, crs=buildings.crs
    )
    s = momepy.courtyard_index(structures_gdf)
    stats = aggregate_stats(s, prefix="courtyard_index")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("courtyard_index", structures_gdf[["geometry"]].assign(courtyard_index=s))
    return cell_gdf


def rectangularity_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute rectangularity (area / area of minimum rotated rectangle).

    Uses equal-area CRS for accurate area computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equal_area
    if buildings.empty:
        return cell_gdf

    s = momepy.rectangularity(buildings)
    stats = aggregate_stats(s, prefix="rectangularity")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("rectangularity", buildings[["geometry"]].assign(rectangularity=s))
    return cell_gdf


def shape_index_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute shape index = sqrt(area/pi) / (0.5 * longest axis). Effectively measures how close the shape is to a circle.

    Dimensionless ratio mixing area and length. Uses equidistant CRS.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    s = momepy.shape_index(buildings)
    stats = aggregate_stats(s, prefix="shape_index")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("shape_index", buildings[["geometry"]].assign(shape_index=s))
    return cell_gdf


def corners_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute corners count (vertices where angle deviates from 180 deg).

    Uses conformal CRS for accurate angle computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.conformal
    if buildings.empty:
        return cell_gdf

    # fails on multipolygons (https://github.com/pysal/momepy/issues/739)
    s = momepy.corners(buildings.explode(ignore_index=True))
    stats = aggregate_stats(s, prefix="corners", include_sum=True)
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("corners", buildings.explode(ignore_index=True)[["geometry"]].assign(corners=s))
    return cell_gdf


def squareness_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute squareness (mean deviation of corner angles from 90 deg).

    Uses conformal CRS for accurate angle computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.conformal
    if buildings.empty:
        return cell_gdf

    # fails on multipolygons (https://github.com/pysal/momepy/issues/739)
    s = momepy.squareness(buildings.explode(ignore_index=True))
    stats = aggregate_stats(s, prefix="squareness")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("squareness", buildings.explode(ignore_index=True)[["geometry"]].assign(squareness=s))
    return cell_gdf


def equivalent_rectangular_index_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute equivalent rectangular index (area and perimeter ratios).

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.

    In building footprint analysis:

    High ERI → rectangular buildings (row houses, slab blocks, industrial sheds)

    Medium ERI → slightly articulated buildings

    Low ERI → complex footprints (L-shapes, courtyards, fragmented forms)
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    s = momepy.equivalent_rectangular_index(buildings)
    stats = aggregate_stats(s, prefix="equivalent_rectangular_index")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("equivalent_rectangular_index", buildings[["geometry"]].assign(equivalent_rectangular_index=s))
    return cell_gdf


def elongation_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute elongation (shorter/longer side of minimum bounding rectangle).

    Dimensionless ratio. Uses equidistant CRS.

    For building footprints:

    High elongation (low ratio) → row houses, slabs, industrial strips

    Low elongation (near 1) → compact blocks, detached houses
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    s = momepy.elongation(buildings)
    stats = aggregate_stats(s, prefix="elongation")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("elongation", buildings[["geometry"]].assign(elongation=s))
    return cell_gdf


def facade_ratio_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute facade ratio (area / perimeter) for raw buildings and for unioned structures.

    Returns two sets of metrics:
    - facade_ratio_*: from raw building geometries.
    - facade_ratio_structure_*: from dissolved (union-all then explode) structures,
      so adjacent buildings are treated as one, matching courtyard metrics.

    Dimensionless ratio mixing area and perimeter. Uses equidistant CRS.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    # Raw building metrics
    s_raw = momepy.facade_ratio(buildings)
    stats_raw = aggregate_stats(s_raw, prefix="facade_ratio")
    for k, v in stats_raw.items():
        cell_gdf[k] = v

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
        cell_gdf[k] = v

    if ctx.dump_dir is not None:
        ctx.dump("facade_ratio", buildings[["geometry"]].assign(facade_ratio=s_raw))
        ctx.dump("facade_ratio_structure", structures_gdf[["geometry"]].assign(facade_ratio=s_struct))

    return cell_gdf


def fractal_dimension_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
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
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    # Raw building metrics
    s_raw = momepy.fractal_dimension(buildings)
    stats_raw = aggregate_stats(s_raw, prefix="fractal_dimension")
    for k, v in stats_raw.items():
        cell_gdf[k] = v

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
        cell_gdf[k] = v

    if ctx.dump_dir is not None:
        ctx.dump("fractal_dimension", buildings[["geometry"]].assign(fractal_dimension=s_raw))
        ctx.dump("fractal_dimension_structure", structures_gdf[["geometry"]].assign(fractal_dimension=s_struct))

    return cell_gdf


def form_factor_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute form factor = surface / volume^(2/3).

    Uses equal-area CRS for accurate surface area and volume computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equal_area
    if buildings.empty:
        return cell_gdf

    s = momepy.form_factor(buildings, buildings["height"])
    stats = aggregate_stats(s, prefix="form_factor")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("form_factor", buildings[["geometry"]].assign(form_factor=s))
    return cell_gdf


def compactness_weighted_axis_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute compactness-weighted axis (longest axis length + compactness) for raw buildings and unioned structures.

    Returns two sets of metrics:
    - compactness_weighted_axis_*: from raw building geometries.
    - compactness_weighted_axis_structure_*: from dissolved (union-all then explode) structures,
      so adjacent buildings are treated as one, matching courtyard metrics.

    Uses equidistant CRS for accurate distance computation.
    Measures how efficiently a polygon fills space relative to its principal axes.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    # Raw building metrics
    s_raw = momepy.compactness_weighted_axis(buildings)
    stats_raw = aggregate_stats(s_raw, prefix="compactness_weighted_axis")
    for k, v in stats_raw.items():
        cell_gdf[k] = v

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
        cell_gdf[k] = v

    if ctx.dump_dir is not None:
        ctx.dump("compactness_weighted_axis", buildings[["geometry"]].assign(compactness_weighted_axis=s_raw))
        ctx.dump("compactness_weighted_axis_structure", structures_gdf[["geometry"]].assign(compactness_weighted_axis=s_struct))

    return cell_gdf


def centroid_corner_distance_metrics(ctx: CellContext) -> gpd.GeoDataFrame:
    """Compute centroid-corner distance (mean and std of distances) for raw buildings and unioned structures.

    Returns two sets of metrics:
    - centroid_corner_distance_*: from raw building geometries.
    - centroid_corner_distance_structure_*: from dissolved (union-all then explode) structures,
      so adjacent buildings are treated as one, matching courtyard metrics.

    Uses equidistant CRS for accurate distance computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    if buildings.empty:
        return cell_gdf

    # Raw building metrics
    # fails on multipolygons (https://github.com/pysal/momepy/issues/739)
    df_raw = momepy.centroid_corner_distance(buildings.explode(ignore_index=True))
    for col in ["mean", "std"]:
        stats = aggregate_stats(df_raw[col], prefix=f"centroid_corner_distance_{col}")
        for k, v in stats.items():
            cell_gdf[k] = v

    # Union and explode into structures (same as courtyard metrics)
    dissolved = buildings.union_all()
    if dissolved.geom_type == "MultiPolygon":
        structures = list(dissolved.geoms)
    else:
        structures = [dissolved]
    structures_gdf = gpd.GeoDataFrame(geometry=structures, crs=buildings.crs)

    # fails on multipolygons (https://github.com/pysal/momepy/issues/739)
    df_struct = momepy.centroid_corner_distance(structures_gdf.explode(ignore_index=True))
    for col in ["mean", "std"]:
        stats = aggregate_stats(
            df_struct[col], prefix=f"centroid_corner_distance_structure_{col}"
        )
        for k, v in stats.items():
            cell_gdf[k] = v

    if ctx.dump_dir is not None:
        _exploded = buildings.explode(ignore_index=True)
        ctx.dump("centroid_corner_distance", _exploded[["geometry"]].assign(
            ccd_mean=df_raw["mean"], ccd_std=df_raw["std"]
        ))
        _struct_exp = structures_gdf.explode(ignore_index=True)
        ctx.dump("centroid_corner_distance_structure", _struct_exp[["geometry"]].assign(
            ccd_mean=df_struct["mean"], ccd_std=df_struct["std"]
        ))

    return cell_gdf
