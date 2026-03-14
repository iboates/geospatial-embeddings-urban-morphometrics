"""Metrics capturing the relationship between buildings and streets."""

import geopandas as gpd
import momepy

from ._utils import CellContext, aggregate_stats


def street_profile_metrics(
    ctx: CellContext,
    distance: float = 10,
    tick_length: float = 50,
) -> gpd.GeoDataFrame:
    """Compute street profile metrics (width, openness, height-width ratio).

    Neighbourhood-aware: uses neighbourhood buildings and highways for the profile
    computation. Guards on focal buildings being empty (no streets to profile).

    Uses equidistant CRS for accurate width/distance computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    if ctx.focal_buildings.equidistant.empty:
        return cell_gdf

    profile_buildings = ctx.neighbourhood_buildings.equidistant
    highways = ctx.neighbourhood_highways.equidistant

    if highways.empty:
        return cell_gdf

    height = profile_buildings["height"] if "height" in profile_buildings.columns else None
    df = momepy.street_profile(
        highways, profile_buildings, distance=distance, tick_length=tick_length, height=height
    )

    summable_cols = {"width", "height"}
    for col in df.columns:
        stats = aggregate_stats(
            df[col].dropna(),
            prefix=f"street_profile_{col}",
            include_sum=col in summable_cols,
        )
        for k, v in stats.items():
            cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("street_profile", highways[["geometry"]].assign(**{
            col: df[col] for col in df.columns
        }))
    return cell_gdf


def nearest_street_distance_metrics(
    ctx: CellContext,
    max_distance: float = 200,
) -> gpd.GeoDataFrame:
    """Compute distance to nearest street.

    Uses focal buildings and neighbourhood highways for accurate edge distances.

    Uses equidistant CRS for accurate distance computation.
    """
    cell_gdf = ctx.focal_buildings.cell_gdf.copy()
    buildings = ctx.focal_buildings.equidistant
    highways = ctx.neighbourhood_highways.equidistant

    if buildings.empty or highways.empty:
        return cell_gdf

    dist_series = buildings.geometry.apply(
        lambda g: highways.geometry.distance(g).min()
    )
    stats = aggregate_stats(dist_series.dropna(), prefix="nearest_street_distance")
    for k, v in stats.items():
        cell_gdf[k] = v
    if ctx.dump_dir is not None:
        ctx.dump("nearest_street_distance", buildings[["geometry"]].assign(nearest_street_distance=dist_series))
    return cell_gdf
