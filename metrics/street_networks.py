"""Network filtering by travel mode (vehicle, pedestrian, all)."""

import pandas as pd

# OSM highway types traversable by each mode.
# Vehicle: roads for motor traffic (excludes pedestrian-only).
VEHICLE_HIGHWAY = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "service", "living_street", "road", "track",
}

# Pedestrian: pedestrian-only infrastructure (including shared roeads)
PEDESTRIAN_HIGHWAY = {
    "unclassified", "residential",
    "service", "living_street", "road", "track",
    "footway", "pedestrian", "path", "steps", "corridor", "bridleway",
}


def _normalize_highway_value(val) -> str | None:
    """Extract highway type from value (handles 'highway=footway' or plain 'footway')."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    if "=" in s:
        s = s.split("=", 1)[-1]
    return s


def _get_highway_series(highways_gdf):
    """Get a Series of highway type values, handling both exploded and compact tag formats."""
    if "highway" in highways_gdf.columns:
        return highways_gdf["highway"].apply(_normalize_highway_value)
    if "tags" in highways_gdf.columns:
        return highways_gdf["tags"].apply(
            lambda t: _normalize_highway_value(t.get("highway") if isinstance(t, dict) else None)
        )
    return None


def filter_highways_by_mode(highways_gdf, mode: str):
    """Filter highways to those traversable by the given mode.

    Args:
        highways_gdf: GeoDataFrame with highway geometries. Expects either a "highway"
            column (quackosm exploded format) or "tags" dict column.
        mode: One of "vehicle", "pedestrian", "all".

    Returns:
        Filtered GeoDataFrame.
    """
    if highways_gdf.empty:
        return highways_gdf

    if mode == "all":
        return highways_gdf.copy()

    highway_series = _get_highway_series(highways_gdf)
    if highway_series is None:
        return highways_gdf.copy()

    allowed = {
        "vehicle": VEHICLE_HIGHWAY,
        "pedestrian": PEDESTRIAN_HIGHWAY,
    }[mode]

    mask = highway_series.apply(lambda hw: hw in allowed if hw else False)
    return highways_gdf[mask].copy()


def split_highways(highways_gdf):
    """Split highways into vehicle and pedestrian networks.

    Returns:
        Tuple of (vehicles_gdf, pedestrians_gdf).
    """
    return (
        filter_highways_by_mode(highways_gdf, "vehicle"),
        filter_highways_by_mode(highways_gdf, "pedestrian"),
    )
