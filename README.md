# Augmentation of Geospatial Embedding Models with Urban Morphometrics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20314975.svg)](https://doi.org/10.5281/zenodo.20314975)

The core of this repository is an **embedding benchmark** (`urban_morphometrics/embedding/`) that asks whether urban morphometrics improve geospatial region embeddings. It evaluates a range of embedders (Hex2Vec, Count/Contextual-Count) on SRAI downstream regression datasets (e.g. house prices, crime), and compares them on a common regression head. See **[Embedding benchmark](#embedding-benchmark)** below, or the detailed [embedding README](urban_morphometrics/embedding/README.md).

The morphometrics pipeline (`urban_morphometrics/`) is the supporting component: it computes per-cell urban form metrics from OpenStreetMap data — building dimensions, shape, spatial distribution, street relationships, and street network connectivity — which the morphometrics-based embedders consume as features.

## Installation

```bash
poetry install
```

`.env` (loaded via python-dotenv) supplies the `WANDB_API_KEY` used by the benchmark runners.

To jump straight to the benchmark, see [Embedding benchmark](#embedding-benchmark). The sections immediately below document the **morphometrics pipeline** — the component that produces the metric features used by the morphometrics-based embedders.

## Running the morphometrics pipeline

### Command line

```bash
python run.py STUDY_AREA PBF_PATH RUN_NAME OUTPUT_FOLDER [options]
```

Or after install:

```bash
urban-morphometrics STUDY_AREA PBF_PATH RUN_NAME OUTPUT_FOLDER [options]
```

**Positional arguments:**

| Argument | Description |
|---|---|
| `STUDY_AREA` | Path to study area file (GeoPackage / GeoJSON / Shapefile) in WGS84 with a `region_id` column or index |
| `PBF_PATH` | Path or URL to a `.pbf` OSM extract. URLs are downloaded once and cached in `OUTPUT_FOLDER/pbf_cache/` |
| `RUN_NAME` | Name for this run, used to organise output folders |
| `OUTPUT_FOLDER` | Root directory for all outputs |

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `--neighbourhood-distance METRES` | 500 | Buffer (m) around each cell for neighbourhood context |
| `--num-quantiles N` | 10 | Number of quantile bands for per-building metrics |
| `--metrics LIST` | `all` | Comma-separated metric names, or `all` |
| `--equal-area-crs CRS` | `EPSG:3395` | CRS for area-preserving projections |
| `--equidistant-crs CRS` | `EPSG:4087` | CRS for distance-preserving projections |
| `--conformal-crs CRS` | `EPSG:3857` | CRS for shape-preserving projections |
| `--debug` | off | Dump intermediate OSM layers to `OUTPUT_FOLDER/RUN_NAME/debug/` |
| `--no-cache` | off | Recompute every cell, ignoring and not writing the metric cache |
| `--metric-config PATH` | — | Path to a JSON config file (see below) |

**Example:**

```bash
urban-morphometrics study_area.gpkg city.osm.pbf run_2024 outputs/ \
    --neighbourhood-distance 300 \
    --metrics degree,meshedness_global,floor_area \
    --metric-config config.json
```

### Python API

```python
import geopandas as gpd
from urban_morphometrics.main import compute_urban_morphometrics

study_area = gpd.read_file("study_area.gpkg").set_index("region_id")

compute_urban_morphometrics(
    study_area_gdf=study_area,
    pbf_path="city.osm.pbf",
    run_name="my_run",
    output_folder="outputs/",
    neighbourhood_distance=500,
    num_quantiles=10,
    metrics=["floor_area", "degree", "meshedness_global"],  # or None for all
    metric_config={"knn_k": 10, "network_subgraph_radius": 3},
)
```

## Output

Results are written to `OUTPUT_FOLDER/RUN_NAME/results/metrics.gpkg` — a GeoPackage with one row per study area cell and one column per metric statistic. For distributed metrics the columns follow the pattern `{metric}_{stat}` where `stat` is `mean`, `median`, `std`, or `q{percentile}` (e.g. `floor_area_mean`, `degree_vehicle_q90`). Graph-level metrics produce a single column per network type (e.g. `meshedness_global_vehicle`).

Intermediate data (per-cell projected GeoDataFrames, tessellations) is cached to `OUTPUT_FOLDER/RUN_NAME/cache/{region_id}/` as Parquet files so the pipeline can resume after interruption.

## Metric config

Computational parameters can be tuned via a JSON file passed with `--metric-config` (CLI) or as a dict via `metric_config=` (API). All keys are optional; omitted keys use the defaults shown below.

```json
{
    "knn_k": 15,
    "tessellation_buffer": "adaptive",
    "tessellation_min_buffer": 0.0,
    "tessellation_max_buffer": 100.0,
    "tessellation_shrink": 0.4,
    "tessellation_segment": 0.5,
    "street_alignment_max_distance": 500.0,
    "network_subgraph_radius": 5,
    "street_profile_distance": 10.0,
    "street_profile_tick_length": 50.0
}
```

| Parameter | Default | Description |
|---|---|---|
| `knn_k` | 15 | Nearest neighbours for KNN graphs (alignment, building_adjacency, mean_interbuilding_distance) |
| `tessellation_buffer` | `"adaptive"` | Clip buffer for Voronoi tessellation. `"adaptive"` uses momepy's Gabriel-graph heuristic; pass a float for a fixed distance in metres |
| `tessellation_min_buffer` | 0.0 | Minimum clip buffer (m) when `tessellation_buffer` is `"adaptive"` |
| `tessellation_max_buffer` | 100.0 | Maximum clip buffer (m) when `tessellation_buffer` is `"adaptive"` |
| `tessellation_shrink` | 0.4 | Negative buffer (m) applied to each building before Voronoi to create a gap between cells |
| `tessellation_segment` | 0.5 | Maximum point spacing (m) during polygon discretisation; smaller = more precise but slower |
| `street_alignment_max_distance` | 500.0 | Maximum search radius (m) for nearest-street matching. Set to `null` for no limit |
| `network_subgraph_radius` | 5 | Hop radius for local subgraph metrics: meshedness, gamma, edge_node_ratio, cyclomatic, mean_node_degree |
| `street_profile_distance` | 10.0 | Spacing (m) between perpendicular ticks when measuring street profiles |
| `street_profile_tick_length` | 50.0 | Length (m) of each tick; should exceed the widest expected street canyon |

## Metric inventory

Full details in [METRICS.md](METRICS.md). Summary:

### Dimension metrics (5)

| Metric | What it measures |
|---|---|
| `courtyard_area` | Area of interior courtyards per dissolved structure |
| `floor_area` | Footprint × floor count |
| `longest_axis_length` | Diameter of minimum bounding circle |
| `perimeter_wall` | Perimeter of joined (touching) structures |
| `volume` | Footprint × building height |

### Shape metrics (15)

`circular_compactness`, `square_compactness`, `convexity`, `courtyard_index`, `rectangularity`, `shape_index`, `corners`, `squareness`, `equivalent_rectangular_index`, `elongation`, `facade_ratio`, `fractal_dimension`, `form_factor`, `compactness_weighted_axis`, `centroid_corner_distance`

### Distribution metrics (9) — require neighbourhood context

| Metric | What it measures |
|---|---|
| `orientation` | Deviation of building axis from cardinal directions |
| `shared_walls` | Shared wall length and ratio with adjacent buildings |
| `alignment` | Orientation consistency among KNN neighbours |
| `neighbor_distance` | Mean distance to Delaunay-triangulation neighbours |
| `mean_interbuilding_distance` | Mean distance to Delaunay+KNN cluster |
| `building_adjacency` | Fraction of KNN neighbours sharing a wall |
| `neighbors` | Number of queen-contiguous tessellation neighbours |
| `cell_alignment` | Deviation between building and tessellation cell orientation |
| `street_alignment` | Deviation between building orientation and nearest street |

### Intensity and street relationship metrics (3) — require neighbourhood context

| Metric | What it measures |
|---|---|
| `courtyards` | Courtyards in each building's contiguous group |
| `street_profile` | Street width, openness, and height-to-width ratio |
| `nearest_street_distance` | Distance from building centroid to nearest street |

### Street connectivity — per-node (11, × vehicle + pedestrian = 22 column groups)

Each metric produces aggregated columns for both `_vehicle` and `_pedestrian` networks.

| Metric | What it measures |
|---|---|
| `degree` | Number of edges at each intersection |
| `clustering` | Clustering coefficient (local triangle density) |
| `mean_node_dist` | Mean edge length at each node |
| `meshedness` | Local circuit density (radius-configurable subgraph) |
| `mean_node_degree` | Mean degree within local subgraph |
| `gamma` | Local edge density vs maximum possible |
| `edge_node_ratio` | Local edges-to-nodes ratio |
| `cyclomatic` | Number of independent circuits in local subgraph |
| `closeness_centrality` | Inverse mean shortest-path distance to all nodes |
| `betweenness_centrality` | Fraction of shortest paths passing through each node |
| `straightness_centrality` | Ratio of Euclidean to network distance |

### Street connectivity — graph-level (4, × vehicle + pedestrian = scalar pairs)

| Metric | Columns | What it measures |
|---|---|---|
| `meshedness_global` | `meshedness_global_{vehicle\|pedestrian}` | Global circuit density |
| `gamma_global` | `gamma_global_{vehicle\|pedestrian}` | Global edge density |
| `cds_length_total` | `cds_length_total_{vehicle\|pedestrian}` | Total cul-de-sac length (m) |
| `proportion` | `proportion_{three\|four\|dead}_{vehicle\|pedestrian}` | Share of 3-way, 4-way, and dead-end nodes |

## Embedding benchmark

The `urban_morphometrics/embedding/` subpackage is the main entry point of the project. Each experiment loads an SRAI dataset, regionalises it to H3, builds an embedder, embeds the regions, trains a PyTorch-Lightning regression head on a downstream target, evaluates it with `HexRegressionEvaluator`, and logs results to Weights & Biases. Running the same regression head over different embedders lets you compare how much signal each embedding carries — and whether adding morphometrics helps.

### Running

```bash
# Run a single experiment from a config (--res / --ds optionally override the config)
python -m urban_morphometrics.embedding.run_experiment \
    --config urban_morphometrics/embedding/configs/hex2vec.yaml --res 9 --ds HouseSalesInKingCounty

# Run every config × seeds × datasets × resolutions
python -m urban_morphometrics.embedding.run_all

# Captum feature attribution for a trained model
python -m urban_morphometrics.embedding.run_captum
```

`run_all.py` discovers every `configs/*.yaml` (except `base.yaml`) and runs each across the seeds, datasets, and resolutions defined at the top of that file.

### Embedders and datasets

Embedders and datasets are registered in factory modules, so adding one means appending a registry entry and (optionally) a small construction/region-building branch:

- **Embedders** (`embedders/embedder_factory.py`): `Hex2Vec`, `GeoVex`, `CountEmbedder`, `ContextualCountEmbedder`, plus the morphometrics-based `UrbanMorphometricsEmbedder`, `Hex2VecUrbanMorphometrics`, and `ContextualUrbanMorphometricsEmbedder`.
- **Datasets** (`data/dataset_factory.py`): `HouseSalesInKingCounty`, `ChicagoCrime`, `PhiladelphiaCrime`, `AirbnbMulticity`.

### Configuration

Experiments are YAML configs in `configs/`, each **merged on top of `configs/base.yaml`**, so a config only lists what differs from the defaults. Top-level keys cover `resolution`, the OSM/morphometrics feature filters, and the neighbourhood radius; the `dataset`, `embedder`, `model`, and `training` sections configure the run.

For the full guide to adding datasets, embedders, and configs, see the **[embedding benchmark README](urban_morphometrics/embedding/README.md)**.

## Running tests

```bash
pytest tests/
```
