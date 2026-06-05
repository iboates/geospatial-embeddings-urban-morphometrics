"""
Thin wrapper around the SRAI HexRegressionEvaluator for test-set prediction
and result persistence.  Keeps run_experiment.py clean.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def predict(
    trainer: pl.Trainer,
    model: nn.Module,
    dataloader: DataLoader,
) -> list:
    """Run inference over a DataLoader and return (h3_indexes, predictions)."""
    model.eval()
    all_preds = []
    for preds in trainer.predict(model, dataloaders=dataloader):
        all_preds.extend(preds.cpu().numpy())

    return all_preds


def evaluate_and_save(
    evaluator: Any,
    dataset: Any,
    predictions: list,
    region_ids: list,
    output_dir: Path,
    experiment_name: str,
    config_path: str,
) -> dict:
    """
    Evaluate on the test set, merge with dev metrics, save results.json, and
    return the full results dict.
    """
    test_results = evaluator.evaluate(
        dataset=dataset,
        predictions=predictions,
        region_ids=region_ids,
        log_metrics=False,
    )
    logger.info("Test results: %s", test_results)

    results = {
        "experiment_name": experiment_name,
        "config_path": config_path,
        "test_results": test_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    return results
