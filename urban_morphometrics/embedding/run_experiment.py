"""
Experiment entry point.

Usage:
    python run_experiment.py --config configs/experiments/hex2vec_king_county.yaml
    python run_experiment.py --config configs/experiments/geovex_king_county.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import MinMaxScaler
from srai.benchmark import HexRegressionEvaluator
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import H3Regionalizer
from torch.utils.data import DataLoader

import wandb
from urban_morphometrics.embedding.config import load_config
from urban_morphometrics.embedding.data.dataset_factory import (
    build_full_regions,
    load_dataset,
)
from urban_morphometrics.embedding.data.preparation import (
    aggregate_per_hex,
    assign_h3_index,
    build_hf_dataset,
    fit_transform_scaler,
    merge_embeddings_with_targets,
    transform_scaler,
)
from urban_morphometrics.embedding.embedders.embedder_factory import build_embedder
from urban_morphometrics.embedding.embedders.pipeline import (
    get_morpho_filter,
    get_osm_filter,
    run_embedding_pipeline,
)
from urban_morphometrics.embedding.evaluator import evaluate_and_save, predict
from urban_morphometrics.embedding.models.regression import (
    RegressionBaseModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(
    config_path: str, resolution: int | None = None, dataset_name: str | None = None
) -> dict:
    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # ── Build experiment name ────────────────────────────────────────────────
    ds_cfg = cfg["dataset"]
    if resolution is None:
        resolution: int = cfg["resolution"]
    if dataset_name is not None:
        ds_cfg["name"] = dataset_name

    cfg_sweep_name = cfg.get("sweep_name", "")
    cfg_exp_name = cfg.get("experiment_name", "")
    exp_name = f"{cfg_sweep_name}_{cfg_exp_name}_{ds_cfg['name']}_r{resolution}"

    logger.info("=== Experiment: %s ===", exp_name)

    output_dir = Path(cfg["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = load_dataset(ds_cfg["name"], str(resolution))
    train_gdf, dev_gdf = dataset.train_test_split(
        test_size=ds_cfg["dev_size"], validation_split=True
    )
    _, test_gdf = dataset.load(version=str(resolution)).values()

    # ── Regionalisation ──────────────────────────────────────────────────────
    regionalizer = H3Regionalizer(resolution=resolution)

    joined_train, regions_train = assign_h3_index(train_gdf, regionalizer)
    joined_dev, regions_dev = assign_h3_index(dev_gdf, regionalizer)
    joined_test, regions_test = assign_h3_index(test_gdf, regionalizer)

    full_regions = build_full_regions(
        regions_train, regionalizer, ds_cfg["name"], train_gdf
    )

    # ── Numerical scaling (optional) ─────────────────────────────────────────
    use_numerical: bool = ds_cfg["use_numerical_columns"]
    numerical_cols = dataset.numerical_columns if use_numerical else []
    scaler = None

    if use_numerical:
        joined_train, scaler = fit_transform_scaler(joined_train, numerical_cols)
        joined_dev = transform_scaler(joined_dev, numerical_cols, scaler)
        joined_test = transform_scaler(joined_test, numerical_cols, scaler)

    # ── Per-hex target (+ numerical) aggregation (average for price and count for crime activity) ────────────────────────────────
    aggregation = ds_cfg["aggregation"]
    aggregated_train = aggregate_per_hex(
        joined_train, dataset.target, numerical_cols, use_numerical, aggregation
    )
    aggregated_dev = aggregate_per_hex(
        joined_dev, dataset.target, numerical_cols, use_numerical, aggregation
    )
    aggregated_test = aggregate_per_hex(
        joined_test, dataset.target, numerical_cols, use_numerical, aggregation
    )

    if aggregation == "count":
        # Could be moved somewhere else
        count_scaler = MinMaxScaler()
        aggregated_train["count"] = count_scaler.fit_transform(
            aggregated_train[["count"]]
        )
        aggregated_dev["count"] = count_scaler.transform(aggregated_dev[["count"]])
        aggregated_dev["count"] = np.clip(aggregated_dev["count"], 0, 1)
        aggregated_test["count"] = count_scaler.transform(aggregated_test[["count"]])
        aggregated_test["count"] = np.clip(aggregated_test["count"], 0, 1)

    # ── Embedder ─────────────────────────────────────────────────────────────
    emb_cfg = cfg["embedder"]
    osm_filter = get_osm_filter(cfg["osm_filter"])
    morpho_filter = get_morpho_filter(cfg["morpho_filter"])
    embedder_name = emb_cfg["name"]
    if "Contextual" in embedder_name:
        neighbourhood = H3Neighbourhood(full_regions)
    else:
        neighbourhood = None
    embedder = build_embedder(
        name=embedder_name,
        hidden_sizes=emb_cfg.get("hidden_sizes", []),
        osm_filter=osm_filter,
        morpho_filter=morpho_filter,
        neighbourhood=neighbourhood,
        neighbourhood_radius=cfg["neighbourhood_radius"],
    )

    embedding_logger = WandbLogger(
        name=exp_name, project="Urban Morphometrics - Embedding"
    )

    fit_kwargs = emb_cfg.get("fit_kwargs", {})

    fit_kwargs.setdefault("trainer_kwargs", {}).update({"logger": embedding_logger})

    morpho_cfg = cfg.get("morphometrics", {})

    emb_train, emb_dev, emb_test = run_embedding_pipeline(
        embedder=embedder,
        embedder_name=emb_cfg["name"],
        exp_name=exp_name,
        regions_train=regions_train,
        regions_dev=regions_dev,
        regions_test=regions_test,
        full_regions=full_regions,
        osm_filter=osm_filter,
        neighbourhood_radius=cfg["neighbourhood_radius"],
        fit_kwargs=fit_kwargs,
        morpho_cfg=morpho_cfg,
    )
    wandb.finish()

    # ── Merge embeddings with targets ────────────────────────────────────────
    merged_train, feat_cols = merge_embeddings_with_targets(
        emb_train, aggregated_train, dataset.target
    )
    merged_dev, _ = merge_embeddings_with_targets(
        emb_dev, aggregated_dev, dataset.target
    )
    merged_test, _ = merge_embeddings_with_targets(
        emb_test, aggregated_test, dataset.target
    )

    # ── PyTorch datasets & loaders ───────────────────────────────────────────
    train_cfg = cfg["training"]
    batch_size: int = train_cfg["batch_size"]

    hf_train = build_hf_dataset(merged_train, feat_cols, dataset.target)
    hf_dev = build_hf_dataset(merged_dev, feat_cols, dataset.target)
    hf_test = build_hf_dataset(merged_test, feat_cols, dataset.target)

    train_loader = DataLoader(hf_train, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(hf_dev, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(hf_test, batch_size=batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    embedding_size: int = hf_train["X"].shape[1]
    model_cfg = cfg["model"]
    model = RegressionBaseModel(
        embeddings_size=embedding_size,
        linear_sizes=model_cfg["linear_sizes"],
        dropout_p=model_cfg.get("dropout_p", 0.2),
        loss_name=train_cfg["loss"],
    )
    evaluator = HexRegressionEvaluator()

    # ── Train ────────────────────────────────────────────────────────────────
    regression_logger = WandbLogger(
        name=exp_name, project="Urban Morphometrics - Regressor"
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename=f"{exp_name}_best_model",
            monitor="val_loss",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=train_cfg["epochs"],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=regression_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader)

    # ── Test evaluation ──────────────────────────────────────────────────────
    h3_indexes = list(merged_test["h3"].values)
    all_preds = predict(
        trainer=trainer,
        model=model,
        dataloader=test_loader,
    )

    results = evaluate_and_save(
        evaluator=evaluator,
        dataset=dataset,
        predictions=all_preds,
        region_ids=h3_indexes,
        output_dir=output_dir,
        experiment_name=exp_name,
        config_path=str(config_path),
    )

    regression_logger.log_metrics(results)

    wandb.finish()

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an SRAI benchmark experiment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config (e.g. configs/experiments/hex2vec_king_county.yaml)",
    )
    parser.add_argument(
        "--res",
        required=False,
        default=None,
        help="Optional resolution (8, 9 or 10) used to overwrite the config",
    )
    parser.add_argument(
        "--ds",
        required=False,
        default=None,
        help="Optional dataset name to overwrite the config",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
