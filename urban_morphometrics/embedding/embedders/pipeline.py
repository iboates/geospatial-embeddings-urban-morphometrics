"""
Handles the full embedding workflow:
  1. Load OSM features for train / dev / test regions
  2. Fit the embedder on train data (if required)
  3. Transform all three splits into embedding DataFrames

Separating this from the main run script keeps things reusable and testable.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from srai.constants import GEOMETRY_COLUMN
from srai.h3 import ring_buffer_h3_regions_gdf
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood

from urban_morphometrics.embedding.data.preparation import get_dataset_crs
from urban_morphometrics.embedding.embedders.embedder_factory import requires_fit
from urban_morphometrics.embedding.filters import ALL_FILTER, EMPTY_FILTER
from urban_morphometrics.main import compute_urban_morphometrics

logger = logging.getLogger(__name__)

# ── OSM filter registry ────────────────────────────────────────────────────────
OSM_FILTER_REGISTRY: dict[str, Any] = {
    "HEX2VEC_FILTER": HEX2VEC_FILTER,
    "EMPTY": EMPTY_FILTER,
}

MORPHO_FILTER_REGISTRY: dict[str, Any] = {
    "ALL_FILTER": ALL_FILTER,
    # ← Add more filters here
}


def get_osm_filter(name: str) -> Any:
    if name not in OSM_FILTER_REGISTRY:
        raise ValueError(
            f"Unknown OSM filter '{name}'. "
            f"Available: {list(OSM_FILTER_REGISTRY.keys())}"
        )
    return OSM_FILTER_REGISTRY[name]


def get_morpho_filter(name: str) -> Any:
    if name not in MORPHO_FILTER_REGISTRY:
        raise ValueError(
            f"Unknown OSM filter '{name}'. "
            f"Available: {list(MORPHO_FILTER_REGISTRY.keys())}"
        )
    raw_morpho_filter = MORPHO_FILTER_REGISTRY[name]
    morpho_filter = []
    for _, metrics in raw_morpho_filter.items():
        morpho_filter.extend(metrics)
    return morpho_filter


def run_embedding_pipeline(
    embedder,
    embedder_name: str,
    exp_name: str,
    regions_train: gpd.GeoDataFrame,
    regions_dev: gpd.GeoDataFrame,
    regions_test: gpd.GeoDataFrame,
    full_regions: gpd.GeoDataFrame,
    osm_filter: Any,
    neighbourhood_radius: int,
    fit_kwargs: dict,
    morpho_cfg: dict,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Run the full embed pipeline for train / dev / test splits.

    Returns:
        (embeddings_train, embeddings_dev, embeddings_test) as GeoDataFrames
        with an added `h3` column holding the region index.
    """
    loader = OSMPbfLoader()
    joiner = IntersectionJoiner()

    # Buffer regions
    buf_train = ring_buffer_h3_regions_gdf(regions_train, neighbourhood_radius)
    buf_dev = ring_buffer_h3_regions_gdf(regions_dev, neighbourhood_radius)
    buf_test = ring_buffer_h3_regions_gdf(regions_test, neighbourhood_radius)

    combined = gpd.GeoDataFrame(
        pd.concat([buf_train, buf_dev, buf_test], ignore_index=False), crs=buf_train.crs
    )

    combined = combined[~combined.index.duplicated(keep="first")]

    logger.info("Loading OSM features for combined region (single pass)...")

    # Load OSM features
    if osm_filter:
        logger.info("Loading OSM features for all regions...")
        osm_all = loader.load(combined, osm_filter)

        logger.info("Joining OSM features for train regions...")
        joint_train = joiner.transform(buf_train, osm_all)

        logger.info("Joining OSM features for dev regions...")
        joint_dev = joiner.transform(buf_dev, osm_all)

        logger.info("Joining OSM features for test regions...")
        joint_test = joiner.transform(buf_test, osm_all)
    else:
        osm_all = gpd.GeoDataFrame(columns=[GEOMETRY_COLUMN])

        joint_train = gpd.GeoDataFrame(columns=[GEOMETRY_COLUMN])
        joint_dev = gpd.GeoDataFrame(columns=[GEOMETRY_COLUMN])
        joint_test = gpd.GeoDataFrame(columns=[GEOMETRY_COLUMN])

    morpho_kwargs = {}

    if morpho_cfg:
        # Compute Urban Morphometrics
        logger.info("Loading Urban Morphometrics features for all regions...")
        cache_name = "_".join(exp_name.split("_")[-2:])
        dataset_name = exp_name.split("_")[-2]
        logger.debug(f"Cache Folder: {cache_name} | Dataset Name: {dataset_name}")
        equal_area_crs, equidistant_crs, conformal_crs = get_dataset_crs(dataset_name)
        morpho_all = compute_urban_morphometrics(
            study_area_gdf=combined,
            pbf_path=None,
            run_name=cache_name,
            output_folder="urban_morphometrics",
            neighbourhood_distance=morpho_cfg["neighbourhood_distance"],
            num_quantiles=morpho_cfg["num_quantiles"],
            equal_area_crs=equal_area_crs,
            equidistant_crs=equidistant_crs,
            conformal_crs=conformal_crs,
            n_workers=20,
            use_cache=True,
        )

        # Get morpho feature columns (exclude geometry)
        morpho_feature_cols = [col for col in morpho_all.columns if col != "geometry"]

        # Join morpho features to train/dev/test splits
        logger.info("Joining morpho features to train/dev/test splits...")
        morpho_train = buf_train.join(morpho_all[morpho_feature_cols])

        # Fit MinMaxScaler on train morpho features
        logger.info("Fitting MinMaxScaler on train morpho features...")
        scaler = MinMaxScaler()
        morpho_train_features = morpho_train[morpho_feature_cols].copy()
        scaler.fit(morpho_train_features)

        # Apply scaling to all splits
        logger.info("Scaling morpho features for all splits...")
        morpho_all_scaled = morpho_all.copy()
        morpho_all_scaled[morpho_feature_cols] = scaler.transform(
            morpho_all[morpho_feature_cols]
        )

        morpho_kwargs["morpho_features_gdf"] = morpho_all_scaled

    # Fit (if needed)
    if requires_fit(embedder_name):
        neighbourhood = H3Neighbourhood(buf_train)
        logger.info("Fitting embedder '%s'...", embedder_name)
        _fit_kwargs = dict(fit_kwargs)  # shallow copy
        if "trainer_kwargs" in _fit_kwargs:
            _fit_kwargs["trainer_kwargs"] = {
                **_fit_kwargs["trainer_kwargs"],
            }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedder.fit(
                regions_gdf=buf_train,
                features_gdf=osm_all,
                joint_gdf=joint_train,
                neighbourhood=neighbourhood,
                **_fit_kwargs,
                **morpho_kwargs,
            )
    else:
        logger.info("Embedder '%s' does not require fitting — skipping.", embedder_name)

    # Transform
    def _transform(buf, osm, joint, label, morpho_kwargs):
        logger.info("Transforming %s split...", label)
        emb = embedder.transform(
            regions_gdf=buf,
            features_gdf=osm,
            joint_gdf=joint,
            **morpho_kwargs,
        )
        emb["h3"] = emb.index
        return emb

    emb_train = _transform(buf_train, osm_all, joint_train, "train", morpho_kwargs)
    emb_dev = _transform(buf_dev, osm_all, joint_dev, "dev", morpho_kwargs)
    emb_test = _transform(buf_test, osm_all, joint_test, "test", morpho_kwargs)

    return emb_train, emb_dev, emb_test
