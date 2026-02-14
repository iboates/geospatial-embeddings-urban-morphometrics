# Urban Morphometrics

Extract OpenStreetMap data for polygons and compute building morphology and street network metrics using [momepy](https://momepy.org/) and [quackosm](https://github.com/kraina-ai/quackosm).

## Installation

```bash
poetry install
```

## Usage

Use from a Jupyter notebook or Python script:

```python
import geopandas as gpd
from main import compute_urban_morphometrics

# Load your polygons (H3 cells, admin boundaries, custom zones, etc.)
polygons = gpd.read_file("my_areas.geojson")

result = compute_urban_morphometrics(
    polygons_gdf=polygons,
    pbf_url="https://download.geofabrik.de/europe/monaco-latest.osm.pbf",
)
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `polygons_gdf` | GeoDataFrame of polygons to compute metrics for (one row per analysis area) |
| `pbf_url` | URL to the OSM PBF file (basename used for cached copy) |
| `save_data_path` | If provided, save buildings, networks, landuse, and output metrics to parquet files in this directory |
| `metrics` | List of metric names to compute (e.g. `["courtyard_area", "elongation"]`). If omitted, all metrics are computed |

**Example with selective metrics and saving:**

```python
result = compute_urban_morphometrics(
    polygons_gdf=polygons,
    pbf_url="https://download.geofabrik.de/europe/monaco-latest.osm.pbf",
    save_data_path="./output",
    metrics=["courtyard_area", "elongation", "degree_vehicle"],
)
```

## What it does

1. **Loads OSM data** – Uses quackosm to extract buildings, highways, and landuse from the PBF, clipped to each polygon.
2. **Splits networks** – Filters highways into vehicle (motor traffic) and pedestrian (footways, paths, etc.) networks.
3. **Computes metrics** – For each polygon, runs a suite of momepy-backed morphology and connectivity metrics:

   - **Building metrics**: courtyard area, floor area, volume, compactness, elongation, corners, orientation, adjacency, alignment, street profile, etc.
   - **Street connectivity**: degree, meshedness, gamma, centrality, and related stats for vehicle and pedestrian networks. Vehicle network uses a directed graph with one-way handling.

4. **Returns results** – A GeoDataFrame with one row per polygon and all metric columns.

Pipeline uses a cached PBF if one exists with the same basename as the URL; otherwise it downloads the file.
