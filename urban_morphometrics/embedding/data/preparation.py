"""
Helpers that sit between raw dataset splits and final PyTorch Datasets:
  - assign H3 hex indexes to each point
  - optionally scale numerical columns
  - aggregate features per hex
  - merge embeddings with targets
  - build HuggingFace / PyTorch datasets
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from srai.regionalizers import H3Regionalizer

logger = logging.getLogger(__name__)


# ── Use proper CRS ────────────────────────────────────────────────────────────

CRS_MAPPING = {
    # Equal Area, Equidistant, Conformal
    "HouseSalesInKingCounty": ("EPSG:5070", "ESRI:102005", "EPSG:2926"),
    "ChicagoCrime": ("EPSG:5070", "ESRI:102005", "EPSG:32616"),
    "PhiladelphiaCrime": ("EPSG:5070", "ESRI:102005", "EPSG:32618"),
    # ← Add more datasets here
}


def get_dataset_crs(dataset_name):
    crs = CRS_MAPPING[dataset_name]
    logger.info(
        f"Using following CRS: Equal Area: {crs[0]} | Equidistant: {crs[1]} | Conformal {crs[2]}"
    )
    return crs


# ── H3 assignment ─────────────────────────────────────────────────────────────


def assign_h3_index(
    gdf: gpd.GeoDataFrame,
    regionalizer: H3Regionalizer,
) -> gpd.GeoDataFrame:
    """Spatial-join points to H3 regions and add a `h3_index` column."""
    gdf_copy = gdf.copy()
    regions = regionalizer.transform(gdf_copy)
    joined = gpd.sjoin(gdf_copy, regions, how="left", predicate="within")
    joined.rename(columns={"region_id": "h3_index"}, inplace=True)
    return joined, regions


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_per_hex(
    joined: gpd.GeoDataFrame,
    target_col: str,
    numerical_cols: list[str],
    use_numerical: bool,
    aggregation: str,
) -> gpd.GeoDataFrame:
    """
    Group by h3_index and return mean of target (+ optionally numerical) cols.
    """
    if aggregation == "average":
        cols = ([target_col] + numerical_cols) if use_numerical else [target_col]
        return joined.groupby("h3_index")[cols].mean()
    elif aggregation == "count":
        return joined.groupby("h3_index").size().reset_index(name="count")
    else:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. Use 'average' or 'count'."
        )


# ── Scaling ───────────────────────────────────────────────────────────────────


def fit_transform_scaler(
    train_joined: gpd.GeoDataFrame,
    numerical_cols: list[str],
) -> tuple[gpd.GeoDataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_joined = train_joined.copy()
    train_joined[numerical_cols] = scaler.fit_transform(train_joined[numerical_cols])
    return train_joined, scaler


def transform_scaler(
    gdf: gpd.GeoDataFrame,
    numerical_cols: list[str],
    scaler: StandardScaler,
) -> gpd.GeoDataFrame:
    gdf_copy = gdf.copy()
    gdf_copy[numerical_cols] = scaler.transform(gdf[numerical_cols])
    return gdf_copy


# ── Merge embeddings with targets ─────────────────────────────────────────────


def merge_embeddings_with_targets(
    embeddings: gpd.GeoDataFrame,
    averages_hex: gpd.GeoDataFrame,
    target_col: str,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """
    Inner-join embeddings to per-hex target averages.

    Returns:
        (merged_gdf, feature_columns)  where feature_columns excludes h3 & target.
    """
    merged = embeddings.merge(
        averages_hex,
        how="inner",
        left_on="region_id",
        right_on="h3_index",
    )
    feature_cols = [
        col for col in merged.columns if col not in (["h3", "h3_index"] + [target_col])
    ]
    return merged, feature_cols


# ── PyTorch dataset assembly ──────────────────────────────────────────────────


def _concat_row(row) -> np.ndarray:
    return np.concatenate([np.atleast_1d(v) for v in row.values]).astype(np.float32)


def build_hf_dataset(
    merged: gpd.GeoDataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Dataset:
    """Build a HuggingFace Dataset with X, X_h3_idx, and y tensors."""
    ds = Dataset.from_dict(
        {
            "X": merged[feature_cols].apply(_concat_row, axis=1).values,
            "X_h3_idx": merged["h3"].values,
            "y": merged[target_col].values,
        }
    )
    ds.set_format(type="torch", columns=["X", "X_h3_idx", "y"])
    return ds
