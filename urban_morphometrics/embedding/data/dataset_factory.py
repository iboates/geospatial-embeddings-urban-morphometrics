"""
Centralised registry for SRAI benchmark datasets.  Adding a new dataset means
adding one entry to DATASET_REGISTRY and (optionally) a custom
`build_full_regions` function.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
from srai.regionalizers import H3Regionalizer

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────
# Maps the `dataset.name` config key → lazy import path + class name.
# Add new datasets here without touching any other file.
DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "HouseSalesInKingCounty": (
        "srai.datasets",
        "HouseSalesInKingCountyDataset",
    ),
    "AirbnbMulticity": (
        "srai.datasets",
        "AirbnbMulticityDataset",
    ),
    "ChicagoCrime": (
        "srai.datasets",
        "ChicagoCrimeDataset",
    ),
    "PhiladelphiaCrime": (
        "srai.datasets",
        "PhiladelphiaCrimeDataset",
    ),
    # ← Add more datasets here
}


def load_dataset(name: str, version: str) -> Any:
    """Instantiate a dataset by registry key and load splits."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    module_path, class_name = DATASET_REGISTRY[name]
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    dataset = cls()
    dataset.load(version=version)
    logger.info("Loaded dataset '%s' (version=%s)", name, version)
    return dataset


def build_full_regions(
    regions_train: gpd.GeoDataFrame,
    regionalizer: H3Regionalizer,
    dataset_name: str,
    train_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame covering the full spatial extent of the training data.

    For single-city datasets a simple convex-hull + buffer approach works.
    For multi-city datasets (e.g. AirbnbMulticity) we regionalise per city to
    avoid loading a continent's worth of OSM data.
    """
    MULTI_CITY_DATASETS = {"AirbnbMulticity"}

    if dataset_name in MULTI_CITY_DATASETS:
        return _build_full_regions_multi_city(train_gdf, regionalizer)

    # Default: single bounding region
    full_geometry = regions_train.union_all().buffer(0.1)
    full_regions = regionalizer.transform(
        gpd.GeoDataFrame(["full"], geometry=[full_geometry]).set_crs(regions_train.crs)
    )
    return full_regions


def _build_full_regions_multi_city(
    train_gdf: gpd.GeoDataFrame,
    regionalizer: H3Regionalizer,
) -> gpd.GeoDataFrame:
    """Per-city full-region computation to bound RAM usage."""
    import pandas as pd

    full_regions_list: list[gpd.GeoDataFrame] = []
    for city, group in train_gdf.groupby("city"):
        logger.info(f"Processing {city}")
        regions_city = regionalizer.transform(group.copy())
        full_geometry = regions_city.union_all().buffer(0.1)
        full_region_gdf = gpd.GeoDataFrame(
            {"city": [city]},
            geometry=[full_geometry],
            crs=regions_city.crs,
        )
        city_regions = regionalizer.transform(full_region_gdf)
        city_regions["city"] = city
        full_regions_list.append(city_regions)

    return gpd.GeoDataFrame(pd.concat(full_regions_list, ignore_index=True))
