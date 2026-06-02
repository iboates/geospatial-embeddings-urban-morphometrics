"""Per-cell spatial context with lazy computation and Parquet-backed caching.

CellContext prepares all projected and filtered GeoDataFrames needed to compute
metrics for a single study area cell. Each property is computed once and cached
to disk so the pipeline can resume after interruption.
"""

import logging
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union

from urban_morphometrics.constants import VEHICLE_HIGHWAY_TYPES, PEDESTRIAN_HIGHWAY_TYPES
from urban_morphometrics.height import resolve_heights
from urban_morphometrics.metric_config import MetricConfig
from urban_morphometrics.oneway import apply_oneway
from urban_morphometrics.osm_loader import OsmData

log = logging.getLogger(__name__)


def _dissolve_buildings(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve touching buildings into superstructures (one row per structure)."""

    dissolved = unary_union(buildings.geometry)
    geoms = list(dissolved.geoms) if hasattr(dissolved, "geoms") else [dissolved]
    return gpd.GeoDataFrame(
        geometry=[g for g in geoms if isinstance(g, Polygon)],
        crs=buildings.crs,
    )


class CellContext:
    """Lazy, cached spatial context for a single study area cell.

    All properties are computed on first access and persisted to Parquet under
    cache_dir. Subsequent accesses (including across pipeline restarts) load
    from the cache instead of recomputing.

    All projected GeoDataFrames use one of the three provided CRS strings:
      - equal-area  (_ea): preserves area; used for most building metrics
      - equidistant (_ed): preserves distance; used for network metrics
      - conformal   (_cf): preserves angles/shape; used for shape metrics

    Neighbourhood properties combine the focal cell's features with features
    from a buffer zone outside the cell to avoid edge effects in graph-based
    metrics. The neighbourhood distance is in metres (applied in equal-area CRS).
    """

    def __init__(
        self,
        region_id,
        cell_geometry,
        osm_data: OsmData,
        neighbourhood_distance: float,
        equal_area_crs: str,
        equidistant_crs: str,
        conformal_crs: str,
        cache_dir: Path,
        config: "MetricConfig | None" = None,
        features_dir: "Path | None" = None,
    ):
        self.region_id = region_id
        self._cell_geometry = cell_geometry
        self._osm_data = osm_data
        self._neighbourhood_distance = neighbourhood_distance
        self._ea_crs = equal_area_crs
        self._ed_crs = equidistant_crs
        self._cf_crs = conformal_crs
        self._cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config if config is not None else MetricConfig()
        self._features_dir = features_dir

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def _load_or_compute(self, name: str, compute_fn) -> gpd.GeoDataFrame:
        path = self._cache_path(name)
        if path.exists():
            result = gpd.read_parquet(path)
        else:
            result = compute_fn()
            result.to_parquet(path)
        if result.index.duplicated().any():
            result = result[~result.index.duplicated(keep="first")]
        return result

    # ------------------------------------------------------------------
    # Internal derived geometries (not cached to disk, cheap to recompute)
    # ------------------------------------------------------------------

    @cached_property
    def _cell_series_wgs84(self) -> gpd.GeoSeries:
        return gpd.GeoSeries([self._cell_geometry], crs="EPSG:4326")

    @cached_property
    def _cell_ea(self):
        return self._cell_series_wgs84.to_crs(self._ea_crs).iloc[0]

    @cached_property
    def _cell_ed(self):
        return self._cell_series_wgs84.to_crs(self._ed_crs).iloc[0]

    @cached_property
    def _cell_buffer_wgs84(self):
        """Cell buffered by neighbourhood_distance, back in WGS84."""
        buffer_ea = gpd.GeoSeries([self._cell_ea], crs=self._ea_crs).buffer(
            self._neighbourhood_distance
        )
        return buffer_ea.to_crs("EPSG:4326").iloc[0]

    @cached_property
    def _focal_building_index(self) -> pd.Index:
        """Index of OSM buildings that intersect this cell."""
        b = self._osm_data.buildings
        return b.index[b.intersects(self._cell_geometry)]

    @cached_property
    def _neighbourhood_building_index(self) -> pd.Index:
        """Index of OSM buildings inside the neighbourhood buffer but not the cell."""
        b = self._osm_data.buildings
        in_buffer = b.intersects(self._cell_buffer_wgs84)
        not_focal = ~b.index.isin(self._focal_building_index)
        return b.index[in_buffer & not_focal]

    @cached_property
    def _focal_highway_index(self) -> pd.Index:
        """Index of OSM highways that intersect this cell."""
        h = self._osm_data.highways
        return h.index[h.intersects(self._cell_geometry)]

    @cached_property
    def _neighbourhood_highway_index(self) -> pd.Index:
        """Index of OSM highways inside the neighbourhood buffer but not the cell."""
        h = self._osm_data.highways
        in_buffer = h.intersects(self._cell_buffer_wgs84)
        not_focal = ~h.index.isin(self._focal_highway_index)
        return h.index[in_buffer & not_focal]

    # ------------------------------------------------------------------
    # Buildings (focal cell)
    # ------------------------------------------------------------------

    @cached_property
    def buildings_ea(self) -> gpd.GeoDataFrame:
        """Focal buildings projected to equal-area CRS."""
        return self._load_or_compute(
            "buildings_ea",
            lambda: self._osm_data.buildings.loc[self._focal_building_index].to_crs(self._ea_crs),
        )

    @cached_property
    def buildings_ed(self) -> gpd.GeoDataFrame:
        """Focal buildings projected to equidistant CRS."""
        return self._load_or_compute(
            "buildings_ed",
            lambda: self._osm_data.buildings.loc[self._focal_building_index].to_crs(self._ed_crs),
        )

    @cached_property
    def buildings_cf(self) -> gpd.GeoDataFrame:
        """Focal buildings projected to conformal CRS."""
        return self._load_or_compute(
            "buildings_cf",
            lambda: self._osm_data.buildings.loc[self._focal_building_index].to_crs(self._cf_crs),
        )

    @cached_property
    def buildings_with_height(self) -> gpd.GeoDataFrame:
        """Focal buildings in equal-area CRS with a resolved numeric 'height' column.

        Height resolution priority: OSM 'height' tag → 'building:levels' * 3 m → 6 m default.
        """
        return self._load_or_compute(
            "buildings_with_height",
            lambda: resolve_heights(self.buildings_ea),
        )

    # ------------------------------------------------------------------
    # Highways (focal cell)
    # ------------------------------------------------------------------

    def _filter_highways(self, types: set, crs: str, index: pd.Index) -> gpd.GeoDataFrame:
        h = self._osm_data.highways.loc[index]
        h = h[h["highway"].isin(types)]
        h = apply_oneway(h)
        return h.to_crs(crs)

    @cached_property
    def vehicle_highways_ea(self) -> gpd.GeoDataFrame:
        """Focal vehicle highways in equal-area CRS with boolean 'oneway' column."""
        return self._load_or_compute(
            "vehicle_highways_ea",
            lambda: self._filter_highways(VEHICLE_HIGHWAY_TYPES, self._ea_crs, self._focal_highway_index),
        )

    @cached_property
    def vehicle_highways_ed(self) -> gpd.GeoDataFrame:
        """Focal vehicle highways in equidistant CRS with boolean 'oneway' column."""
        return self._load_or_compute(
            "vehicle_highways_ed",
            lambda: self._filter_highways(VEHICLE_HIGHWAY_TYPES, self._ed_crs, self._focal_highway_index),
        )

    @cached_property
    def vehicle_highways_cf(self) -> gpd.GeoDataFrame:
        """Focal vehicle highways in conformal CRS with boolean 'oneway' column."""
        return self._load_or_compute(
            "vehicle_highways_cf",
            lambda: self._filter_highways(VEHICLE_HIGHWAY_TYPES, self._cf_crs, self._focal_highway_index),
        )

    @cached_property
    def pedestrian_highways_ea(self) -> gpd.GeoDataFrame:
        """Focal pedestrian highways in equal-area CRS."""
        return self._load_or_compute(
            "pedestrian_highways_ea",
            lambda: self._filter_highways(PEDESTRIAN_HIGHWAY_TYPES, self._ea_crs, self._focal_highway_index),
        )

    @cached_property
    def pedestrian_highways_ed(self) -> gpd.GeoDataFrame:
        """Focal pedestrian highways in equidistant CRS."""
        return self._load_or_compute(
            "pedestrian_highways_ed",
            lambda: self._filter_highways(PEDESTRIAN_HIGHWAY_TYPES, self._ed_crs, self._focal_highway_index),
        )

    @cached_property
    def pedestrian_highways_cf(self) -> gpd.GeoDataFrame:
        """Focal pedestrian highways in conformal CRS."""
        return self._load_or_compute(
            "pedestrian_highways_cf",
            lambda: self._filter_highways(PEDESTRIAN_HIGHWAY_TYPES, self._cf_crs, self._focal_highway_index),
        )

    # ------------------------------------------------------------------
    # Landuse (focal cell)
    # ------------------------------------------------------------------

    @cached_property
    def landuse_ea(self) -> gpd.GeoDataFrame:
        """Focal landuse polygons in equal-area CRS."""
        return self._load_or_compute(
            "landuse_ea",
            lambda: self._osm_data.landuse.loc[
                self._osm_data.landuse.index[
                    self._osm_data.landuse.intersects(self._cell_geometry)
                ]
            ].to_crs(self._ea_crs),
        )

    @cached_property
    def water_ea(self) -> gpd.GeoDataFrame:
        """Focal water polygons (natural=water) in equal-area CRS."""
        return self._load_or_compute(
            "water_ea",
            lambda: self._osm_data.water.loc[
                self._osm_data.water.index[
                    self._osm_data.water.intersects(self._cell_geometry)
                ]
            ].to_crs(self._ea_crs),
        )

    @cached_property
    def pedestrian_areas_ea(self) -> gpd.GeoDataFrame:
        """Focal pedestrian area polygons (highway=pedestrian areas) in equal-area CRS."""
        return self._load_or_compute(
            "pedestrian_areas_ea",
            lambda: self._osm_data.pedestrian_areas.loc[
                self._osm_data.pedestrian_areas.index[
                    self._osm_data.pedestrian_areas.intersects(self._cell_geometry)
                ]
            ].to_crs(self._ea_crs),
        )

    # ------------------------------------------------------------------
    # Neighbourhood layers
    # ------------------------------------------------------------------

    @cached_property
    def neighbourhood_buildings(self) -> gpd.GeoDataFrame:
        """Buildings within neighbourhood_distance of the cell but outside it, in equal-area CRS."""
        return self._load_or_compute(
            "neighbourhood_buildings",
            lambda: self._osm_data.buildings.loc[self._neighbourhood_building_index].to_crs(self._ea_crs),
        )

    @cached_property
    def neighbourhood_vehicle_highways(self) -> gpd.GeoDataFrame:
        """Vehicle highways within neighbourhood_distance of the cell but outside it, in equal-area CRS."""
        return self._load_or_compute(
            "neighbourhood_vehicle_highways",
            lambda: self._filter_highways(VEHICLE_HIGHWAY_TYPES, self._ea_crs, self._neighbourhood_highway_index),
        )

    @cached_property
    def neighbourhood_pedestrian_highways(self) -> gpd.GeoDataFrame:
        """Pedestrian highways within neighbourhood_distance of the cell but outside it, in equal-area CRS."""
        return self._load_or_compute(
            "neighbourhood_pedestrian_highways",
            lambda: self._filter_highways(PEDESTRIAN_HIGHWAY_TYPES, self._ea_crs, self._neighbourhood_highway_index),
        )

    # ------------------------------------------------------------------
    # Dissolved buildings (touching superstructures)
    # ------------------------------------------------------------------

    @cached_property
    def dissolved_buildings_ea(self) -> gpd.GeoDataFrame:
        """Focal buildings dissolved into touching superstructures, in equal-area CRS."""
        return self._load_or_compute(
            "dissolved_buildings_ea",
            lambda: _dissolve_buildings(self.buildings_ea),
        )

    @cached_property
    def dissolved_buildings_cf(self) -> gpd.GeoDataFrame:
        """Focal buildings dissolved into touching superstructures, in conformal CRS."""
        return self.dissolved_buildings_ea.to_crs(self._cf_crs)

    @cached_property
    def dissolved_buildings_ed(self) -> gpd.GeoDataFrame:
        """Focal buildings dissolved into touching superstructures, in equidistant CRS."""
        return self.dissolved_buildings_ea.to_crs(self._ed_crs)

    # ------------------------------------------------------------------
    # Focal + neighbourhood combined (for edge-effect-free metrics)
    # ------------------------------------------------------------------

    @cached_property
    def focal_plus_neighbourhood_buildings(self) -> gpd.GeoDataFrame:
        """Focal and neighbourhood buildings combined, in equal-area CRS."""
        result = pd.concat([self.buildings_ea, self.neighbourhood_buildings])
        if result.index.duplicated().any():
            result = result[~result.index.duplicated(keep="first")]
        return result

    @cached_property
    def focal_plus_neighbourhood_vehicle_highways(self) -> gpd.GeoDataFrame:
        """Focal and neighbourhood vehicle highways combined, in equal-area CRS."""
        result = pd.concat([self.vehicle_highways_ea, self.neighbourhood_vehicle_highways])
        if result.index.duplicated().any():
            result = result[~result.index.duplicated(keep="first")]
        return result

    @cached_property
    def focal_plus_neighbourhood_pedestrian_highways(self) -> gpd.GeoDataFrame:
        """Focal and neighbourhood pedestrian highways combined, in equal-area CRS."""
        result = pd.concat([self.pedestrian_highways_ea, self.neighbourhood_pedestrian_highways])
        if result.index.duplicated().any():
            result = result[~result.index.duplicated(keep="first")]
        return result

    # ------------------------------------------------------------------
    # Shared spatial graphs (in-memory only; not persisted to Parquet)
    # Built on focal + neighbourhood buildings to avoid edge effects.
    # Returned as None when there are too few buildings to form a graph.
    # ------------------------------------------------------------------

    @cached_property
    def knn_graph(self):
        """KNN graph on focal + neighbourhood buildings (k from MetricConfig.knn_k).

        Returns None if there are fewer buildings than k+1, in which case metrics
        that depend on this graph return all-NaN aggregations.
        coplanar='jitter' handles the rare case of coincident building centroids.
        """
        from libpysal.graph import Graph

        all_b = self.focal_plus_neighbourhood_buildings
        if len(all_b) < 2:
            return None
        k = min(self.config.knn_k, len(all_b) - 1)
        return Graph.build_knn(all_b.centroid, k=k, coplanar="jitter")

    @cached_property
    def delaunay_graph(self):
        """Delaunay triangulation graph on focal + neighbourhood buildings.

        Returns None if there are fewer than 3 buildings (minimum for a triangulation).
        coplanar='jitter' handles coincident centroids.
        """
        from libpysal.graph import Graph

        all_b = self.focal_plus_neighbourhood_buildings
        if len(all_b) < 3:
            return None
        return Graph.build_triangulation(all_b.centroid, method="delaunay", coplanar="jitter")

    @cached_property
    def contiguity_graph(self):
        """Rook contiguity graph on focal + neighbourhood buildings.

        Buildings that share a wall segment (or touch within tolerance) are
        considered rook-contiguous. Returns None for empty building sets.
        """
        from libpysal.graph import Graph

        all_b = self.focal_plus_neighbourhood_buildings
        if all_b.empty:
            return None
        try:
            return Graph.build_contiguity(all_b, rook=True)
        except Exception:
            log.warning(
                "Contiguity graph construction failed for region %s", self.region_id, exc_info=True
            )
            return None

    @cached_property
    def tessellation(self):
        """Morphological (Voronoi) tessellation on focal + neighbourhood buildings.

        Parameters (clip, shrink, segment) come from MetricConfig. The result is
        persisted to tessellation.parquet alongside the other cached layers so it
        is not recomputed on subsequent pipeline runs. Returns None if there are
        too few buildings or tessellation fails on degenerate geometry (no file
        is written in that case).
        The tessellation index matches focal_plus_neighbourhood_buildings.index,
        allowing downstream metrics to filter results to focal buildings via reindex.
        """
        import momepy

        path = self._cache_path("tessellation")
        if path.exists():
            return gpd.read_parquet(path)

        all_b = self.focal_plus_neighbourhood_buildings
        if len(all_b) < 3:
            return None
        try:
            try:
                clip = momepy.buffered_limit(
                    all_b,
                    buffer=self.config.tessellation_buffer,
                    min_buffer=self.config.tessellation_min_buffer,
                    max_buffer=self.config.tessellation_max_buffer
                )
            except Exception:
                # buffered_limit uses Gabriel triangulation internally and fails
                # with too few buildings or coincident centroids. Fall back to a
                # uniform buffer; use max_buffer as the numeric distance since
                # tessellation_buffer may be the string "adaptive".
                log.debug(
                    "buffered_limit failed for region %s, using uniform buffer fallback",
                    self.region_id,
                )
                clip = all_b.buffer(self.config.tessellation_max_buffer).union_all()
            result = momepy.morphological_tessellation(
                all_b,
                clip=clip,
                shrink=self.config.tessellation_shrink,
                segment=self.config.tessellation_segment,
            )
        except Exception:
            log.warning(
                "Morphological tessellation failed for region %s", self.region_id, exc_info=True
            )
            return None
        result.to_parquet(path)
        return result

    @cached_property
    def tessellation_queen_graph(self):
        """Queen contiguity graph on the morphological tessellation.

        Returns None if tessellation is unavailable or empty.
        """
        from libpysal.graph import Graph

        tess = self.tessellation
        if tess is None or tess.empty:
            return None
        return Graph.build_contiguity(tess, rook=False)

    # ------------------------------------------------------------------
    # Street network graphs (in-memory only; not persisted to Parquet)
    # Built from focal + neighbourhood highways in equidistant CRS so
    # that edge lengths are accurate for distance-based metrics.
    # ------------------------------------------------------------------

    @cached_property
    def vehicle_graph(self):
        """Directed primal NetworkX graph of focal + neighbourhood vehicle highways.

        Nodes are street intersections; edges are segments with length ('mm_len')
        and direction respecting the 'oneway' column. Returns None if there are
        no vehicle streets in the neighbourhood.
        """
        from urban_morphometrics.street_graph import build_vehicle_graph

        streets = self.focal_plus_neighbourhood_vehicle_highways.to_crs(self._ed_crs)
        return build_vehicle_graph(streets, save_dir=self._features_dir, tolerance=self.config.graph_endpoint_snap_tolerance)

    @cached_property
    def pedestrian_graph(self):
        """Undirected primal NetworkX graph of focal + neighbourhood pedestrian highways.

        All segments are treated as bidirectional. Returns None if there are no
        pedestrian streets in the neighbourhood.
        """
        from urban_morphometrics.street_graph import build_pedestrian_graph

        streets = self.focal_plus_neighbourhood_pedestrian_highways.to_crs(self._ed_crs)
        return build_pedestrian_graph(streets, save_dir=self._features_dir, tolerance=self.config.graph_endpoint_snap_tolerance)
