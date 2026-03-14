"""Utility functions for metrics computation."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
from libpysal.graph import Graph
from pyproj import CRS
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


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


@dataclass
class PreparedBuildings:
    """Buildings projected into CRSes suited for different geometric operations.

    Attributes:
        equal_area: GeoDataFrame projected to an equal-area CRS (accurate area).
        equidistant: GeoDataFrame projected to an equidistant CRS (accurate distances).
        conformal: GeoDataFrame projected to a conformal CRS (accurate angles).
        cell_gdf: Cell polygon GeoDataFrame (projected to equidistant CRS).
    """

    equal_area: gpd.GeoDataFrame
    equidistant: gpd.GeoDataFrame
    conformal: gpd.GeoDataFrame
    cell_gdf: gpd.GeoDataFrame


@dataclass
class PreparedHighways:
    """Highways projected into CRSes suited for different geometric operations.

    Attributes:
        equidistant: GeoDataFrame projected to an equidistant CRS (accurate distances).
        conformal: GeoDataFrame projected to a conformal CRS (accurate angles).
    """

    equidistant: gpd.GeoDataFrame
    conformal: gpd.GeoDataFrame


def _estimate_fallback_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Estimate a UTM CRS from a GeoDataFrame, with a fallback."""
    try:
        return gdf.estimate_utm_crs()
    except Exception:
        return CRS.from_epsg(32632)


def prepare_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> PreparedBuildings:
    """Prepare buildings for metrics: filter to polygons, reproject to three CRSes.

    Each CRS is used for the geometric operations it's best suited for:
    - equal_area_crs: area calculations (courtyard area, floor area, volume, etc.)
    - equidistant_crs: distance/length calculations (perimeter, axis length, etc.)
    - conformal_crs: angular calculations (orientation, squareness, corners, etc.)

    Area and height columns are added to all three projections. Area is always
    computed from the equal-area projection for accuracy.

    Args:
        buildings_gdf: Raw buildings GeoDataFrame (WGS84).
        cell_polygon: The analysis cell polygon (WGS84).
        equal_area_crs: CRS for area calculations. Defaults to estimated UTM.
        equidistant_crs: CRS for distance calculations. Defaults to estimated UTM.
        conformal_crs: CRS for angular calculations. Defaults to estimated UTM.

    Returns:
        PreparedBuildings with equal_area, equidistant, conformal, and cell_gdf.
    """
    empty_gdf = gpd.GeoDataFrame()

    if buildings_gdf.empty:
        cell_gdf = gpd.GeoDataFrame(
            [cell_polygon], columns=["geometry"], geometry="geometry", crs=4326
        )
        fallback = _estimate_fallback_crs(cell_gdf)
        cell_gdf = cell_gdf.to_crs(equidistant_crs or fallback)
        return PreparedBuildings(
            equal_area=empty_gdf,
            equidistant=empty_gdf,
            conformal=empty_gdf,
            cell_gdf=cell_gdf,
        )

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
        fallback = _estimate_fallback_crs(cell_gdf)
        cell_gdf = cell_gdf.to_crs(equidistant_crs or fallback)
        return PreparedBuildings(
            equal_area=empty_gdf,
            equidistant=empty_gdf,
            conformal=empty_gdf,
            cell_gdf=cell_gdf,
        )

    # Resolve CRS defaults
    fallback = _estimate_fallback_crs(buildings)
    ea_crs = CRS(equal_area_crs) if equal_area_crs is not None else fallback
    ed_crs = CRS(equidistant_crs) if equidistant_crs is not None else fallback
    cf_crs = CRS(conformal_crs) if conformal_crs is not None else fallback

    # Resolve building height: prefer pre-parsed 'height' column, then
    # fall back to building_levels * 3m, then default 6m (≈2 storeys).
    if "height" not in buildings.columns:
        # Legacy path: try inferring from tags dict
        if "tags" in buildings.columns:
            buildings["height"] = buildings["tags"].apply(_infer_height_from_tags)
        else:
            buildings["height"] = np.nan

    # Where height is still missing, derive from building_levels
    if "building_levels" in buildings.columns:
        missing_height = buildings["height"].isna()
        has_levels = buildings["building_levels"].notna()
        buildings.loc[missing_height & has_levels, "height"] = (
            buildings.loc[missing_height & has_levels, "building_levels"] * 3.0
        )

    buildings["height"] = buildings["height"].fillna(6.0)

    # Ensure building_levels column exists (for floor_area_metrics)
    if "building_levels" not in buildings.columns:
        buildings["building_levels"] = np.nan

    # Project to equal-area first for accurate area computation
    buildings_ea = buildings.to_crs(ea_crs)
    area_values = buildings_ea.geometry.area
    buildings_ea["area"] = area_values

    # Project to equidistant and conformal, carrying area from equal-area
    buildings_ed = buildings.to_crs(ed_crs)
    buildings_ed["area"] = area_values.values

    buildings_cf = buildings.to_crs(cf_crs)
    buildings_cf["area"] = area_values.values

    cell_gdf = cell_gdf.to_crs(ed_crs)

    return PreparedBuildings(
        equal_area=buildings_ea,
        equidistant=buildings_ed,
        conformal=buildings_cf,
        cell_gdf=cell_gdf,
    )


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
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
) -> PreparedHighways:
    """Prepare highways for metrics: filter to LineStrings, reproject to two CRSes.

    - equidistant_crs: distance/length calculations (connectivity, profile width).
    - conformal_crs: angular calculations (street orientation/alignment).

    Args:
        highways_gdf: Raw highways GeoDataFrame (WGS84).
        cell_polygon: The analysis cell polygon (WGS84).
        equidistant_crs: CRS for distance calculations. Defaults to estimated UTM.
        conformal_crs: CRS for angular calculations. Defaults to estimated UTM.

    Returns:
        PreparedHighways with equidistant and conformal projections.
    """
    empty_gdf = gpd.GeoDataFrame()

    if highways_gdf.empty:
        return PreparedHighways(equidistant=empty_gdf, conformal=empty_gdf)

    if highways_gdf.crs is None:
        highways_gdf = highways_gdf.set_crs(4326)
    highways = highways_gdf.copy()

    line_mask = highways.geom_type.isin(["LineString", "MultiLineString"])
    highways = highways[line_mask].copy()

    if highways.empty:
        return PreparedHighways(equidistant=empty_gdf, conformal=empty_gdf)

    # Resolve CRS defaults
    cell_temp = gpd.GeoDataFrame(
        [cell_polygon], columns=["geometry"], geometry="geometry", crs=4326
    )
    fallback = _estimate_fallback_crs(cell_temp)
    ed_crs = CRS(equidistant_crs) if equidistant_crs is not None else fallback
    cf_crs = CRS(conformal_crs) if conformal_crs is not None else fallback

    highways_ed = highways.to_crs(ed_crs)
    highways_cf = highways.to_crs(cf_crs)

    return PreparedHighways(equidistant=highways_ed, conformal=highways_cf)


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


# ---------------------------------------------------------------------------
# Internal graph-building helpers (used by CellContext lazy properties)
# ---------------------------------------------------------------------------

def _build_tessellation_graphs(
    buildings: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame | None, Graph | None, Graph | None]:
    """Build morphological tessellation, queen-contiguity, and KNN graphs.

    Returns (tessellation, contiguity, knn) — any may be None on failure.
    Built from neighbourhood buildings so edge cells have complete graphs.
    """
    if buildings.empty:
        return None, None, None

    try:
        limit = momepy.buffered_limit(buildings, buffer=5)
        tessellation = momepy.morphological_tessellation(buildings, clip=limit)
    except Exception:
        return None, None, None

    try:
        contiguity = Graph.build_contiguity(tessellation, rook=False)
    except Exception:
        contiguity = None

    try:
        knn = Graph.build_knn(buildings.centroid, k=min(15, len(buildings)))
    except Exception:
        knn = None

    return tessellation, contiguity, knn


def _add_oneway_to_highways(highways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a boolean ``oneway`` column to a prepared highways GeoDataFrame."""
    highways = highways.copy()
    has_tags = "tags" in highways.columns
    has_highway = "highway" in highways.columns
    oneway = []
    for i in range(len(highways)):
        tags = highways["tags"].iloc[i] if has_tags else None
        hw = highways["highway"].iloc[i] if has_highway else None
        if hw is None and tags and isinstance(tags, dict):
            hw = tags.get("highway")
        oneway.append(_parse_oneway(tags, hw))
    highways["oneway"] = oneway
    return highways


def _build_street_graph(
    highways: gpd.GeoDataFrame,
    directed: bool = False,
) -> nx.MultiGraph | nx.MultiDiGraph | None:
    """Build a NetworkX street graph from prepared (equidistant) highways.

    Args:
        highways: Projected highways GeoDataFrame (equidistant CRS).
        directed: If True, build a directed graph (vehicle network with oneway).

    Returns:
        NetworkX graph, or None if the graph cannot be built.
    """
    if highways.empty or len(highways) < 2:
        return None
    try:
        cleaned = momepy.remove_false_nodes(highways)
        if cleaned.empty:
            return None
        if directed:
            cleaned = _add_oneway_to_highways(cleaned)
            G = momepy.gdf_to_nx(
                cleaned, length="mm_len", directed=True, oneway_column="oneway"
            )
        else:
            G = momepy.gdf_to_nx(cleaned, length="mm_len", directed=False)
        if G.number_of_edges() == 0:
            return None
        return G
    except Exception as e:
        logger.debug("Street graph creation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# CellContext — shared pre-computed data for a single cell
# ---------------------------------------------------------------------------

@dataclass
class CellContext:
    """Pre-computed data for a single cell, shared across all metrics.

    Contains focal data (buildings/highways within the cell boundary) and
    neighbourhood data (focal + surrounding cells). Shared spatial graphs are
    computed lazily on first access and cached for reuse across metrics.

    Construct via :func:`build_cell_context` rather than directly.

    Attributes:
        focal_buildings: Buildings in the focal cell projected to three CRSes.
        focal_highways: All highways in the focal cell projected to two CRSes.
        neighbourhood_buildings: Buildings in focal + neighbouring cells.
        neighbourhood_highways: All highways in focal + neighbouring cells.
        focal_vehicles: Vehicle network in the focal cell.
        focal_pedestrians: Pedestrian network in the focal cell.
        neighbourhood_vehicles: Vehicle network in focal + neighbouring cells.
        neighbourhood_pedestrians: Pedestrian network in focal + neighbouring cells.
    """

    focal_buildings: PreparedBuildings
    focal_highways: PreparedHighways
    neighbourhood_buildings: PreparedBuildings
    neighbourhood_highways: PreparedHighways
    focal_vehicles: PreparedHighways
    focal_pedestrians: PreparedHighways
    neighbourhood_vehicles: PreparedHighways
    neighbourhood_pedestrians: PreparedHighways

    # --- optional debug export ---
    cell_id: str = ""
    dump_dir: Path | None = None

    # --- lazy-cache backing fields (excluded from constructor and repr) ---
    _building_graphs_computed: bool = field(default=False, init=False, repr=False)
    _tessellation: gpd.GeoDataFrame | None = field(default=None, init=False, repr=False)
    _contiguity: Graph | None = field(default=None, init=False, repr=False)
    _knn: Graph | None = field(default=None, init=False, repr=False)
    _vehicle_graph_computed: bool = field(default=False, init=False, repr=False)
    _vehicle_graph: nx.MultiDiGraph | None = field(default=None, init=False, repr=False)
    _pedestrian_graph_computed: bool = field(default=False, init=False, repr=False)
    _pedestrian_graph: nx.MultiGraph | None = field(default=None, init=False, repr=False)

    def dump(self, name: str, gdf: gpd.GeoDataFrame) -> None:
        """Write a GeoDataFrame to {dump_dir}/{name}_{cell_id}.gpkg if dump_dir is set."""
        if self.dump_dir is None or gdf is None or gdf.empty:
            return
        dump_dir = Path(self.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        safe_id = str(self.cell_id).replace("/", "_").replace(":", "_")
        filename = f"{name}_{safe_id}.gpkg" if safe_id else f"{name}.gpkg"
        path = dump_dir / filename
        try:
            gdf.to_file(path, driver="GPKG")
            logger.debug("Dumped %s to %s", name, path)
        except Exception as e:
            logger.warning("Failed to dump %s: %s", name, e)

    def _ensure_building_graphs(self) -> None:
        if not self._building_graphs_computed:
            self._tessellation, self._contiguity, self._knn = (
                _build_tessellation_graphs(self.neighbourhood_buildings.equidistant)
            )
            self._building_graphs_computed = True
            if self._tessellation is not None and self.dump_dir is not None:
                self.dump("tessellation", self._tessellation)

    @property
    def tessellation(self) -> gpd.GeoDataFrame | None:
        """Morphological tessellation built from neighbourhood buildings."""
        self._ensure_building_graphs()
        return self._tessellation

    @property
    def contiguity(self) -> Graph | None:
        """Queen-adjacency graph of tessellation cells (neighbourhood scope)."""
        self._ensure_building_graphs()
        return self._contiguity

    @property
    def knn(self) -> Graph | None:
        """K-nearest-neighbours graph (k≤15) of building centroids (neighbourhood scope)."""
        self._ensure_building_graphs()
        return self._knn

    @property
    def vehicle_graph(self) -> nx.MultiDiGraph | None:
        """Directed NetworkX graph of the neighbourhood vehicle network."""
        if not self._vehicle_graph_computed:
            self._vehicle_graph = _build_street_graph(
                self.neighbourhood_vehicles.equidistant, directed=True
            )
            self._vehicle_graph_computed = True
        return self._vehicle_graph

    @property
    def pedestrian_graph(self) -> nx.MultiGraph | None:
        """Undirected NetworkX graph of the neighbourhood pedestrian network."""
        if not self._pedestrian_graph_computed:
            self._pedestrian_graph = _build_street_graph(
                self.neighbourhood_pedestrians.equidistant, directed=False
            )
            self._pedestrian_graph_computed = True
        return self._pedestrian_graph


def build_cell_context(
    focal_buildings_gdf: gpd.GeoDataFrame,
    focal_highways_gdf: gpd.GeoDataFrame,
    focal_vehicles_gdf: gpd.GeoDataFrame,
    focal_pedestrians_gdf: gpd.GeoDataFrame,
    neighbourhood_buildings_gdf: gpd.GeoDataFrame,
    neighbourhood_highways_gdf: gpd.GeoDataFrame,
    neighbourhood_vehicles_gdf: gpd.GeoDataFrame,
    neighbourhood_pedestrians_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    equal_area_crs: CRS | str | int | None = None,
    equidistant_crs: CRS | str | int | None = None,
    conformal_crs: CRS | str | int | None = None,
    cell_id: str = "",
    dump_dir: Path | None = None,
) -> CellContext:
    """Construct a CellContext by projecting all input GeoDataFrames.

    Calls :func:`prepare_buildings` twice (focal + neighbourhood) and
    :func:`prepare_highways` six times (focal/neighbourhood × highways/vehicles/
    pedestrians), then returns a :class:`CellContext` whose shared graphs are
    computed lazily on first access and cached for reuse.

    Args:
        focal_buildings_gdf: Buildings within the focal cell (WGS84).
        focal_highways_gdf: All highways within the focal cell (WGS84).
        focal_vehicles_gdf: Vehicle network within the focal cell (WGS84).
        focal_pedestrians_gdf: Pedestrian network within the focal cell (WGS84).
        neighbourhood_buildings_gdf: Buildings in focal + adjacent cells (WGS84).
        neighbourhood_highways_gdf: All highways in focal + adjacent cells (WGS84).
        neighbourhood_vehicles_gdf: Vehicle network in focal + adjacent cells (WGS84).
        neighbourhood_pedestrians_gdf: Pedestrian network in focal + adjacent cells (WGS84).
        cell_polygon: The focal cell polygon (WGS84).
        equal_area_crs: Override CRS for area calculations. Defaults to estimated UTM.
        equidistant_crs: Override CRS for distance calculations. Defaults to estimated UTM.
        conformal_crs: Override CRS for angular calculations. Defaults to estimated UTM.

    Returns:
        :class:`CellContext` ready for metric computation.
    """
    crs_kw = dict(
        equal_area_crs=equal_area_crs,
        equidistant_crs=equidistant_crs,
        conformal_crs=conformal_crs,
    )
    hw_crs_kw = dict(
        equidistant_crs=equidistant_crs,
        conformal_crs=conformal_crs,
    )
    ctx = CellContext(
        focal_buildings=prepare_buildings(focal_buildings_gdf, cell_polygon, **crs_kw),
        focal_highways=prepare_highways(focal_highways_gdf, cell_polygon, **hw_crs_kw),
        neighbourhood_buildings=prepare_buildings(neighbourhood_buildings_gdf, cell_polygon, **crs_kw),
        neighbourhood_highways=prepare_highways(neighbourhood_highways_gdf, cell_polygon, **hw_crs_kw),
        focal_vehicles=prepare_highways(focal_vehicles_gdf, cell_polygon, **hw_crs_kw),
        focal_pedestrians=prepare_highways(focal_pedestrians_gdf, cell_polygon, **hw_crs_kw),
        neighbourhood_vehicles=prepare_highways(neighbourhood_vehicles_gdf, cell_polygon, **hw_crs_kw),
        neighbourhood_pedestrians=prepare_highways(neighbourhood_pedestrians_gdf, cell_polygon, **hw_crs_kw),
        cell_id=cell_id,
        dump_dir=dump_dir,
    )
    if dump_dir is not None:
        ctx.dump("focal_buildings", ctx.focal_buildings.equidistant)
        ctx.dump("neighbourhood_buildings", ctx.neighbourhood_buildings.equidistant)
        ctx.dump("focal_highways", ctx.focal_highways.equidistant)
        ctx.dump("neighbourhood_highways", ctx.neighbourhood_highways.equidistant)
    return ctx
