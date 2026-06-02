# Benchmark Experiment Runner

Benchmarks region embeddings on SRAI downstream regression datasets. Each run
builds an embedder, embeds H3 regions, trains a regression head, and evaluates
it with `HexRegressionEvaluator`, logging to Weights & Biases.

All paths below are relative to the repo root. Commands are run as modules so
imports resolve regardless of the working directory.

## Quick Start

```bash
# Single experiment (config is required; --res / --ds optionally override it)
python -m urban_morphometrics.embedding.run_experiment \
  --config urban_morphometrics/embedding/configs/hex2vec.yaml --res 9 --ds HouseSalesInKingCounty

# Every config × seeds × datasets × resolutions
python -m urban_morphometrics.embedding.run_all

# Captum feature attribution for a trained model
python -m urban_morphometrics.embedding.run_captum
```

`run_all.py` discovers every `configs/*.yaml` except `base.yaml` and runs each
across the `RANDOM_SEEDS`, `DATASETS`, and `RESOLUTIONS` defined at the top of
that file. A `WANDB_API_KEY` in `.env` (loaded via python-dotenv) is required
for logging.

Available configs: `hex2vec.yaml`, `hex2vec_only_urban_morpho.yaml`,
`hex2vec_poi_and_urban_morpho.yaml`, `count_embedder.yaml`, `count_urban_morpho.yaml`,
`contextual_count_embedder.yaml`, `contextual_count_urban_morpho.yaml`,
`raw_urban_morpho.yaml`, `contextual_raw_urban_morpho.yaml`.

## Adding a New Dataset

1. Register it in `urban_morphometrics/embedding/data/dataset_factory.py`:

```python
DATASET_REGISTRY["MyNewDataset"] = ("srai.datasets", "MyNewDatasetClass")
```

2. If it is multi-city (like `AirbnbMulticity`), add its key to
   `MULTI_CITY_DATASETS` inside `build_full_regions`.

3. Create an experiment config (see below).

## Adding a New Embedder

1. Register it in `urban_morphometrics/embedding/embedders/embedder_factory.py`:

```python
EMBEDDER_REGISTRY["MyEmbedder"] = ("srai.embedders", "MyEmbedderClass")
```

2. Add a construction branch in `build_embedder()` if the constructor
   signature differs from the existing embedders.

3. If it does not need a `.fit()` call, add its key to `NO_FIT_EMBEDDERS`.

4. Create an experiment config:

```yaml
embedder:
  name: "MyEmbedder"
  hidden_sizes: [128, 64]
  fit_kwargs:
    batch_size: 256
    trainer_kwargs:
      max_epochs: 5
```

Registered embedders: `Hex2Vec`, `GeoVex`, `CountEmbedder`,
`ContextualCountEmbedder`, `UrbanMorphometricsEmbedder`,
`Hex2VecUrbanMorphometrics`, `ContextualUrbanMorphometricsEmbedder`.

---

## Configuration System

Every experiment config is **merged on top of `configs/base.yaml`** (by
`config.load_config`), so you only need to list the values that differ from the
defaults.

Top-level keys:

| Key | Purpose |
|---|---|
| `sweep_name` | Prefix for the generated experiment name and W&B run |
| `output_dir` | Root directory for experiment outputs |
| `resolution` | H3 resolution for regionalisation (overridable via `--res`) |
| `osm_filter` | OSM feature filter name (e.g. `HEX2VEC_FILTER`) — see `filters.py` |
| `morpho_filter` | Morphometrics feature-group filter name (e.g. `ALL_FILTER`) — see `filters.py` |
| `neighbourhood_radius` | Ring-buffer size around each region (used by contextual / GeoVex embedders) |

Sections:

| Section | Purpose |
|---|---|
| `dataset` | Dataset name, numerical columns toggle, dev split size |
| `embedder` | Embedder class, architecture (`hidden_sizes`), `fit_kwargs` |
| `model` | Regression head layer sizes (`linear_sizes`), dropout |
| `training` | Epochs, batch size, learning rate, early-stopping patience |
| `morphometrics` | Morphometrics cache / computation settings for morpho-based embedders |
