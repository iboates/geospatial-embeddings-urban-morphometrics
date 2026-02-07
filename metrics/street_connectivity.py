"""Street network connectivity metrics for vehicle, cycle, and pedestrian networks."""

import logging

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
from shapely.geometry import Polygon

from ._utils import _parse_oneway, aggregate_stats, prepare_highways

logger = logging.getLogger(__name__)

MODES = ("vehicle", "pedestrian")

# Per-node metrics (aggregated with mean, median, std, deciles)
NODE_METRICS = [
    "degree",
    "meshedness",
    "mean_node_dist",
    "mean_node_degree",
    "gamma",
    "edge_node_ratio",
    "cyclomatic",
    "clustering",
    "closeness_centrality",
    "betweenness_centrality",
    "straightness_centrality",
]

# Graph-level metrics (single value per mode)
GRAPH_METRICS = ["meshedness_global", "gamma_global", "cds_length_total"]

# Proportion sub-metrics
PROPORTION_KEYS = ("three", "four", "dead")


def _add_oneway_column(highways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add oneway column for directed graph. True=one-way, False=bidirectional."""
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


def _to_graph(
    highways: gpd.GeoDataFrame,
    directed: bool = False,
    oneway_column: str | None = None,
) -> nx.MultiGraph | nx.MultiDiGraph | None:
    """Convert highways to networkx graph, with preprocessing."""
    if highways.empty or len(highways) < 2:
        return None
    try:
        cleaned = momepy.remove_false_nodes(highways)
        if cleaned.empty:
            return None
        kwargs = {"length": "mm_len", "directed": directed}
        if oneway_column and oneway_column in cleaned.columns:
            kwargs["oneway_column"] = oneway_column
        G = momepy.gdf_to_nx(cleaned, **kwargs)
        if G.number_of_edges() == 0:
            return None
        return G
    except Exception as e:
        logger.debug("Graph creation failed: %s", e)
        return None


def _compute_node_metrics(G: nx.MultiGraph, mode: str) -> dict[str, float]:
    """Compute connectivity metrics and aggregate per-node values.

    mode: "vehicle" or "pedestrian"; keys become degree_vehicle_mean, meshedness_pedestrian_mean, etc.
    """
    out = {}
    if G.number_of_nodes() == 0:
        return out

    try:
        G = momepy.node_degree(G, name="degree")
    except Exception:
        return out

    metric_fns = [
        ("meshedness", lambda g: momepy.meshedness(g, radius=5, verbose=False)),
        ("mean_node_dist", lambda g: momepy.mean_node_dist(g, verbose=False)),
        ("mean_node_degree", lambda g: momepy.mean_node_degree(g, radius=5, verbose=False)),
        ("gamma", lambda g: momepy.gamma(g, radius=5, verbose=False)),
        ("edge_node_ratio", lambda g: momepy.edge_node_ratio(g, radius=5, verbose=False)),
        ("cyclomatic", lambda g: momepy.cyclomatic(g, radius=5, verbose=False)),
        ("clustering", momepy.clustering),
        ("closeness_centrality", lambda g: momepy.closeness_centrality(g, verbose=False)),
        ("betweenness_centrality", lambda g: momepy.betweenness_centrality(g, verbose=False)),
        ("straightness_centrality", lambda g: momepy.straightness_centrality(g, verbose=False)),
    ]
    for name, fn in metric_fns:
        try:
            G = fn(G)
        except Exception as e:
            logger.debug("Metric %s failed: %s", name, e)

    result = momepy.nx_to_gdf(G, lines=False)
    nodes = result[0] if isinstance(result, tuple) else result
    if nodes.empty:
        return out

    for attr in NODE_METRICS:
        if attr not in nodes.columns:
            continue
        try:
            stats = aggregate_stats(
                nodes[attr],
                prefix=f"{attr}_{mode}",
                include_median=True,
                include_sum=(attr == "degree"),
                include_deciles=True,
            )
            out.update(stats)
        except Exception as e:
            logger.debug("Aggregation for %s failed: %s", attr, e)

    for name in GRAPH_METRICS:
        fn = {
            "meshedness_global": lambda: momepy.meshedness(G, radius=None),
            "gamma_global": lambda: momepy.gamma(G, radius=None),
            "cds_length_total": lambda: momepy.cds_length(G, radius=None, mode="sum", verbose=False),
        }[name]
        try:
            val = fn()
            if isinstance(val, (int, float, np.floating)):
                out[f"{name}_{mode}"] = float(val)
        except Exception:
            pass
    try:
        prop = momepy.proportion(
            G, radius=None, three="three", four="four", dead="dead", verbose=False
        )
        if isinstance(prop, dict):
            for k, v in prop.items():
                out[f"proportion_{k}_{mode}"] = float(v)
    except Exception:
        pass

    return out


def connectivity_metrics(
    network_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
    mode: str,
) -> gpd.GeoDataFrame:
    """Compute street connectivity metrics for a pre-filtered network."""
    highways = prepare_highways(network_gdf, cell_polygon)

    cell_gdf = gpd.GeoDataFrame(
        [cell_polygon], columns=["geometry"], geometry="geometry", crs=4326
    )
    if highways.empty:
        return cell_gdf

    if highways.crs is not None:
        cell_gdf = cell_gdf.to_crs(highways.crs)

    # Vehicle network: directed graph with oneway handling (directionality matters)
    directed = mode == "vehicle"
    oneway_col = None
    if directed:
        highways = _add_oneway_column(highways)
        oneway_col = "oneway"

    G = _to_graph(highways, directed=directed, oneway_column=oneway_col)
    if G is None:
        return cell_gdf

    stats = _compute_node_metrics(G, mode=mode)
    return cell_gdf.assign(**stats)


def compute_connectivity_metrics_by_name(
    vehicles_gdf: gpd.GeoDataFrame,
    pedestrians_gdf: gpd.GeoDataFrame,
    cell_polygon: Polygon,
) -> dict[str, gpd.GeoDataFrame]:
    """Compute connectivity metrics for vehicle and pedestrian networks.

    Returns one GeoDataFrame per metric-mode combo (e.g. meshedness_vehicle,
    meshedness_pedestrian, degree_vehicle, degree_pedestrian). All top-level keys.
    """
    cell_gdf = gpd.GeoDataFrame(
        [cell_polygon], columns=["geometry"], geometry="geometry", crs=4326
    )
    all_stats: dict[str, float] = {}

    for mode, gdf in [("vehicle", vehicles_gdf), ("pedestrian", pedestrians_gdf)]:
        result = connectivity_metrics(gdf, cell_polygon, mode=mode)
        for col in result.columns:
            if col != "geometry":
                all_stats[col] = result[col].iloc[0]

    # Group by metric_mode (e.g. degree_vehicle, meshedness_pedestrian) - each gets its own key
    # col format: {metric}_{mode} or {metric}_{mode}_{stat}
    metric_cols: dict[str, dict[str, float]] = {}
    for col, val in all_stats.items():
        for m in MODES:
            suffix = f"_{m}"
            if col == suffix or col.endswith(suffix) or f"{suffix}_" in col:
                # Base is everything up to and including _vehicle or _pedestrian
                if f"{suffix}_" in col:
                    base = col[: col.index(f"{suffix}_") + len(suffix)]
                else:
                    base = col
                if base not in metric_cols:
                    metric_cols[base] = {}
                metric_cols[base][col] = val
                break

    out: dict[str, gpd.GeoDataFrame] = {}
    for metric_name, cols in metric_cols.items():
        if cols:
            gdf = cell_gdf.copy()
            gdf = gdf.assign(**cols)
            out[metric_name] = gdf

    return out
