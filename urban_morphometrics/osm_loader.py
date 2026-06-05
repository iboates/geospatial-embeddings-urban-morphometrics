"""Load OSM data for a study area from a .pbf file using QuackOSM."""

import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from quackosm import convert_geometry_to_geodataframe, convert_pbf_to_geodataframe
from tqdm import tqdm

# QuackOSM spawns worker processes. On Linux/WSL2 the default start method may
# be "spawn", which requires a `if __name__ == "__main__"` guard in the caller.
# Setting it to "fork" avoids that requirement and is safe on Linux.
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("fork")

log = logging.getLogger(__name__)

_POLYGON_TYPES = {"Polygon", "MultiPolygon"}
_LINE_TYPES = {"LineString", "MultiLineString"}


def _keep_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    before = len(gdf)
    gdf = gdf[gdf.geometry.geom_type.isin(_POLYGON_TYPES)].copy()
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"]
    if gdf.index.duplicated().any():
        keep = gdf.geometry.area.groupby(level=0).idxmax()
        gdf = gdf.loc[keep]
    log.debug("  geometry filter: %d -> %d polygons", before, len(gdf))
    return gdf


def _keep_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    before = len(gdf)
    gdf = gdf[gdf.geometry.geom_type.isin(_LINE_TYPES)].copy()
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type == "LineString"]
    if gdf.index.duplicated().any():
        keep = gdf.geometry.length.groupby(level=0).idxmax()
        gdf = gdf.loc[keep]
    log.debug("  geometry filter: %d -> %d linestrings", before, len(gdf))
    return gdf


@dataclass
class OsmData:
    buildings: gpd.GeoDataFrame
    highways: gpd.GeoDataFrame
    landuse: gpd.GeoDataFrame
    water: gpd.GeoDataFrame
    pedestrian_areas: gpd.GeoDataFrame


def load_osm_data(pbf_path: Path | None, study_area_gdf: gpd.GeoDataFrame) -> OsmData:
    """Load buildings, highways, and landuse from a .pbf file clipped to the study area.

    Results are cached automatically by QuackOSM based on the pbf path and filters.

    Args:
        pbf_path: Local path to the .pbf OSM dump.
        study_area_gdf: Study area in WGS84. The union of all polygons is used as
            the geometry filter so only features overlapping the study area are loaded.

    Returns:
        OsmData with buildings, highways, and landuse GeoDataFrames.
    """
    geometry_filter = study_area_gdf.geometry.union_all()

    layers = [
        (
            "buildings",
            {"building": True, "building:levels": True, "height": True},
            _keep_polygons,
        ),
        ("highways", {"highway": True, "oneway": True}, _keep_lines),
        ("landuse", {"landuse": True}, _keep_polygons),
        ("water", {"natural": "water"}, _keep_polygons),
        ("pedestrian_areas", {"highway": "pedestrian"}, _keep_polygons),
    ]

    results = {}
    for name, tags_filter, clean_fn in tqdm(layers, desc="OSM layers", unit="layer"):
        if pbf_path is None:
            log.debug("%s: extracting from study area geometry...", name)
            gdf = convert_geometry_to_geodataframe(
                geometry_filter, tags_filter=tags_filter
            )
        else:
            gdf = convert_pbf_to_geodataframe(
                pbf_path,
                tags_filter=tags_filter,
                geometry_filter=geometry_filter,
                keep_all_tags=False,
            )
        gdf = clean_fn(gdf)
        log.debug("%s: %d features", name, len(gdf))
        results[name] = gdf

    return OsmData(**results)
