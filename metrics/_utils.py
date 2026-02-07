"""Utility functions for metrics computation."""

import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


def _infer_height_from_tags(tags) -> float | None:
    """Extract building height in meters from OSM tags.

    Checks 'height' (with unit parsing) and 'building:levels' (×3m per level).
    """
    if tags is None or not isinstance(tags, dict):
        return None
    # Try explicit height first (e.g. "15", "15m", "50 ft")
    h = tags.get("height")
    if h is not None and isinstance(h, str):
        h = h.strip().lower()
        # Parse numeric value
        match = re.match(r"^([\d.]+)\s*(m|metres?|meters?|ft|feet)?", h)
        if match:
            val = float(match.group(1))
            unit = (match.group(2) or "m").lower()
            if "ft" in unit or "feet" in unit:
                val *= 0.3048
            return val
    elif isinstance(h, (int, float)):
        return float(h)
    # Fallback: building:levels × 3m
    levels = tags.get("building:levels")
    if levels is not None:
        try:
            return float(levels) * 3.0
        except (ValueError, TypeError):
            pass
    return None


def prepare_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Prepare buildings for metrics: filter to polygons, reproject, add height/area.

    Returns (buildings_projected, cell_projected) in a suitable projected CRS.
    """
    if buildings_gdf.empty:
        return buildings_gdf, gpd.GeoDataFrame()

    # Ensure CRS
    if buildings_gdf.crs is None:
        buildings_gdf = buildings_gdf.set_crs(4326)
    buildings = buildings_gdf.copy()

    # Filter to polygons only
    poly_mask = buildings.geom_type.isin(["Polygon", "MultiPolygon"])
    buildings = buildings[poly_mask].copy()

    cell_gdf = gpd.GeoDataFrame(
        [cell_polygon], columns=["geometry"], geometry="geometry", crs=4326
    )

    if buildings.empty:
        try:
            utm_crs = cell_gdf.estimate_utm_crs()
        except Exception:
            utm_crs = "EPSG:32632"
        cell_gdf = cell_gdf.to_crs(utm_crs)
        return buildings, cell_gdf

    # Reproject to UTM for accurate area/length
    try:
        utm_crs = buildings.estimate_utm_crs()
    except Exception:
        utm_crs = "EPSG:32632"  # fallback for Central Europe
    buildings = buildings.to_crs(utm_crs)
    cell_gdf = cell_gdf.to_crs(utm_crs)

    # Add area and height
    buildings["area"] = buildings.geometry.area
    if "tags" in buildings.columns:
        heights = buildings["tags"].apply(_infer_height_from_tags)
        buildings["height"] = heights
    else:
        buildings["height"] = np.nan
    # Default height where missing (e.g. 2 floors)
    buildings["height"] = buildings["height"].fillna(6.0)

    return buildings, cell_gdf


def _parse_oneway(tags, highway_val) -> bool:
    """Parse OSM oneway tag. Returns True if one-way only, False if bidirectional.

    OSM oneway=yes/true/1 -> one-way. oneway=no/false/0 or absent -> two-way.
    highway=motorway and junction=roundabout default to one-way.
    """
    if tags is None or not isinstance(tags, dict):
        return _default_oneway(highway_val, None)
    oneway = tags.get("oneway")
    junction = tags.get("junction")
    if oneway is None:
        return _default_oneway(highway_val, junction)
    val = str(oneway).strip().lower()
    if val in ("yes", "true", "1"):
        return True
    if val in ("no", "false", "0"):
        return False
    if val == "-1":
        return True  # one-way reverse; geometry direction unchanged for now
    return _default_oneway(highway_val, junction)


def _default_oneway(highway_val, junction) -> bool:
    """Default oneway when tag absent: motorway and roundabout are one-way."""
    if junction and str(junction).lower() == "roundabout":
        return True
    if highway_val and str(highway_val).lower() in ("motorway", "motorway_link"):
        return True
    return False


def prepare_highways(
    highways_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """Prepare highways for metrics: filter to LineStrings, reproject."""
    if highways_gdf.empty:
        return highways_gdf

    if highways_gdf.crs is None:
        highways_gdf = highways_gdf.set_crs(4326)
    highways = highways_gdf.copy()

    line_mask = highways.geom_type.isin(["LineString", "MultiLineString"])
    highways = highways[line_mask].copy()

    if highways.empty:
        return highways

    try:
        cell_temp = gpd.GeoDataFrame(
            [cell_polygon], columns=["geometry"], geometry="geometry", crs=4326
        )
        utm_crs = cell_temp.estimate_utm_crs()
    except Exception:
        utm_crs = "EPSG:32632"
    highways = highways.to_crs(utm_crs)
    return highways


def aggregate_stats(
    series: pd.Series,
    prefix: str = "",
    include_deciles: bool = True,
    include_sum: bool = False,
    include_median: bool = False,
) -> dict[str, float]:
    """Aggregate a series into cell-level statistics.

    Returns mean, std, and deciles (10th–90th) where applicable.
    Optionally includes sum for extensive/count metrics, median when requested.
    """
    valid = series.dropna()
    if valid.empty:
        return {}

    out = {}
    p = f"{prefix}_" if prefix else ""
    out[f"{p}mean"] = valid.mean()
    if len(valid) > 1:
        out[f"{p}std"] = valid.std()
    else:
        out[f"{p}std"] = np.nan

    if include_median:
        out[f"{p}median"] = valid.median()

    if include_sum:
        out[f"{p}sum"] = valid.sum()

    if include_deciles and len(valid) >= 2:
        for q in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            out[f"{p}p{q}"] = valid.quantile(q / 100)

    return out
