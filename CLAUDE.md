# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two coupled subsystems:

1. **Urban morphometrics pipeline** (`urban_morphometrics/`) — calculates urban form metrics from OpenStreetMap data. Reads a study area GeoDataFrame (H3 hex cells with `region_id` index), loads OSM buildings and highways via QuackOSM, and computes per-cell aggregated metrics.
2. **Embedding benchmark** (`urban_morphometrics/embedding/`) — benchmarks region embeddings (Hex2Vec, GeoVex, Count, and morphometrics-based embedders) on SRAI downstream regression datasets (house prices, crime). Trains a PyTorch-Lightning regression head per embedder/dataset/resolution and logs to Weights & Biases.

## Commands

```bash
# Install dependencies
poetry install

# --- Morphometrics pipeline ---
# Run the pipeline (CLI)
python -m urban_morphometrics study_area.gpkg pbf_url run_name /output/folder \
  --neighbourhood-distance 500 --num-quantiles 10 --metrics "volume,floor_area" --debug

# Quick test run (Karlsruhe, Germany)
python run.py

# --- Embedding benchmark ---
# Run a single experiment from a config
python -m urban_morphometrics.embedding.run_experiment \
  --config urban_morphometrics/embedding/configs/hex2vec.yaml [--res 9] [--ds HouseSalesInKingCounty]

# Run every config × seeds × datasets × resolutions
python -m urban_morphometrics.embedding.run_all

# Captum-based feature attribution for a trained model
python -m urban_morphometrics.embedding.run_captum
```

`.env` (loaded via python-dotenv) supplies the `WANDB_API_KEY` used by the benchmark runners.

No linter or formatter is configured. Tests live in `tests/` (`pytest tests/`).

## Architecture

### Data Flow (morphometrics pipeline)

Study area GeoDataFrame → `load_osm_data()` → `OsmData(buildings, highways, …)` → per-cell `CellContext` (lazy-computed, Parquet-cached) → `compute_metrics()` → results GeoPackage.

### Key Modules

- **`main.py`** — CLI entry point and `compute_urban_morphometrics()` orchestrator
- **`osm_loader.py`** — QuackOSM PBF loading with download caching
- **`cell_context.py`** — Per-cell spatial context; all properties are `@cached_property` backed by Parquet files for pipeline resumption
- **`height.py`** — Building height resolution: `height` tag → `building:levels × 3m` → default `6m`
- **`metric_config.py`** — Tunable computational parameters (KNN k, tessellation buffers, subgraph radius, …)
- **`street_graph.py` / `oneway.py`** — Street network construction (vehicle + pedestrian graphs) for connectivity metrics
- **`metrics/__init__.py`** — Registry-based dispatch (`@register("name")` decorator) and `compute_metrics()`
- **`metrics/aggregation.py`** — `aggregate_series()` produces `{prefix}_mean`, `_median`, `_std`, `_q10`…`_q100`

### Embedding benchmark (`urban_morphometrics/embedding/`)

- **`run_experiment.py`** — single experiment: load SRAI dataset → H3 regionalise → build embedder → embed regions → train `RegressionBaseModel` head → evaluate with `HexRegressionEvaluator` → log to W&B. CLI flags `--config`, `--res`, `--ds`.
- **`run_all.py`** — discovers every `configs/*.yaml` (except `base.yaml`) and runs it across `RANDOM_SEEDS`, `DATASETS`, and `RESOLUTIONS`.
- **`run_captum.py`** — feature-attribution analysis of a trained regression head.
- **`config.py`** — `load_config()` deep-merges an experiment config on top of `configs/base.yaml`.
- **`data/dataset_factory.py`** — `DATASET_REGISTRY` (config `name` → SRAI dataset class) and `build_full_regions()`; multi-city datasets handled via `MULTI_CITY_DATASETS`.
- **`data/preparation.py`** — H3 assignment, per-hex target aggregation, scaling, HF dataset/loss helpers.
- **`embedders/embedder_factory.py`** — `EMBEDDER_REGISTRY` (config `name` → embedder class), `build_embedder()`, and `NO_FIT_EMBEDDERS` / `requires_fit()`.
- **`embedders/`** — custom morphometrics-based embedders (`UrbanMorphometricsEmbedder`, `Hex2VecUrbanMorphometrics`, `ContextualUrbanMorphometricsEmbedder`) and the shared `pipeline.py`.
- **`filters.py`** — OSM feature filters and the `ALL_FILTER` morphometrics feature groups used as embedding inputs.
- **`models/regression.py`** — `RegressionBaseModel` Lightning module (the benchmark's downstream head).
- **`configs/`** — flat YAML experiment configs (`hex2vec.yaml`, `count_embedder.yaml`, `raw_urban_morpho.yaml`, …), each merged onto `base.yaml`.

### Metric Function Contract

Each metric is a decorated function: `@register("metric_name")` with signature `def compute(ctx: CellContext, num_quantiles: int) -> dict[str, float]`. Returns a flat dict of aggregated statistics.

### Three CRS Strategy

- **Equal-area (EPSG:3395)** — area/volume calculations
- **Equidistant (EPSG:4087)** — distance/network metrics
- **Conformal (EPSG:3857)** — shape metrics (angle-preserving)

CellContext provides buildings pre-projected to each CRS (`buildings_ea`, `buildings_ed`, `buildings_cf`).

### Cache Structure

```
{output_folder}/{run_name}/
├── cache/{region_id}/     # Per-cell Parquet files (buildings, highways, metrics)
├── results/metrics.gpkg   # Final output
└── debug/                 # Optional debug GeoPackages
```

## Implementation Status

The morphometrics pipeline is complete: registered metrics span dimension, shape, distribution, intensity/street-relationship, and street connectivity (per-node + graph-level, for vehicle and pedestrian networks). See `SPECIFICATION.md` and `METRICS.md` for the full metric catalogue.

The embedding benchmark is the active area of work on the `embedding-benchmark` branch. See `urban_morphometrics/embedding/README.md` for how to add datasets, embedders, and experiment configs.
