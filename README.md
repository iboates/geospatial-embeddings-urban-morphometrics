# Urban Morphometrics

Extract OpenStreetMap data for an H3 cell and compute building morphology and street network metrics using [momepy](https://momepy.org/) and [quackosm](https://github.com/kraina-ai/quackosm).

## Installation

```bash
poetry install
```

## Usage

Run from the command line

```bash
poetry run python main.py \
  --lat <lat value> \
  --lon <lon value> \
  --pbf_url <PBF URL>
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `lat`, `lon` | Center point of the H3 cell |
| `pbf_url` | URL to the OSM PBF file (basename used for cached copy) |
| `h3_level` | H3 resolution (0–15), default `8` |
| `return_dict` | If `True`, return a dict of metric name → GeoDataFrame instead of a single merged GeoDataFrame |
| `save_data_path` | If provided, save buildings, vehicles, pedestrians, and landuse to parquet files in this directory |

**Example with options:**

```bash
poetry run python main.py \
  --lat <lat value> \
  --lon <lon value> \
  --pbf_url <PBF URL>
  --h3_level 9 \
  --return_dict=True \
  --save_data_path ./output
```

## What it does

1. **Loads OSM data** – Uses quackosm to extract buildings, highways, and landuse from the PBF for the given H3 cell.
2. **Splits networks** – Filters highways into vehicle (motor traffic) and pedestrian (footways, paths, etc.) networks.
3. **Computes metrics** – Runs a suite of momepy-backed morphology and connectivity metrics:

   - **Building metrics**: courtyard area, floor area, volume, compactness, elongation, corners, orientation, adjacency, alignment, street profile, etc.
   - **Street connectivity**: degree, meshedness, gamma, centrality, and related stats for vehicle and pedestrian networks. Vehicle network uses a directed graph with one-way handling.

4. **Returns results** – A GeoDataFrame with one row (the cell polygon) and all metric columns, or a dict of metric-name → GeoDataFrame when `return_dict=True`.

Pipeline uses a cached PBF if one exists with the same basename as the URL; otherwise it downloads the file.
