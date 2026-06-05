"""Regression head used on top of spatial embeddings."""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
    SymmetricMeanAbsolutePercentageError,
)

# ── Loss registry ─────────────────────────────────────────────────────────────
LOSS_REGISTRY: dict[str, nn.Module] = {
    "SmoothL1": nn.SmoothL1Loss(),
    "MSE": nn.MSELoss(),
    "MAE": nn.L1Loss(),
}


def build_loss(name: str) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name]


def _build_metrics() -> MetricCollection:
    """Return a MetricCollection with all six regression metrics."""
    return MetricCollection(
        {
            "mse": MeanSquaredError(squared=True),
            "rmse": MeanSquaredError(squared=False),
            "mae": MeanAbsoluteError(),
            "mape": MeanAbsolutePercentageError(),
            "smape": SymmetricMeanAbsolutePercentageError(),
            "r2": R2Score(),
        }
    )


class RegressionBaseModel(pl.LightningModule):
    """
    MLP regression head — PyTorch Lightning module.

    Stacks linear layers with ReLU activations and optional dropout
    (every other layer), ending in a single scalar output.
    """

    def __init__(
        self,
        embeddings_size: int,
        linear_sizes: Optional[list[int]] = None,
        activation_function: Optional[nn.Module] = None,
        dropout_p: float = 0.2,
        loss_name: str = "SmoothL1",
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["activation_function"])

        if linear_sizes is None:
            linear_sizes = [500, 1000]
        if activation_function is None:
            activation_function = nn.ReLU()

        self.loss_fn = build_loss(loss_name)

        self.val_metrics = _build_metrics()
        self.test_metrics = _build_metrics()

        self.model = nn.Sequential()
        previous_size = embeddings_size
        for idx, size in enumerate(linear_sizes):
            self.model.add_module(f"linear_{idx}", nn.Linear(previous_size, size))
            self.model.add_module(f"relu_{idx}", activation_function)
            previous_size = size
            if idx % 2:
                self.model.add_module(f"dropout_{idx}", nn.Dropout(p=dropout_p))
        self.model.add_module("linear_out", nn.Linear(previous_size, 1))

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── Training ──────────────────────────────────────────────────────────────
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch["X"], batch["y"]
        batch_size = x.size(0)
        preds = self(x).squeeze(-1)
        loss = self.loss_fn(preds, y.float())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    # ── Validation ────────────────────────────────────────────────────────────
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch["X"], batch["y"]
        batch_size = x.size(0)
        preds = self(x).squeeze(-1)
        y = y.float()
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.val_metrics.update(preds, y)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(self.val_metrics, prefix="val")

    # ── Test ──────────────────────────────────────────────────────────────────
    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch["X"], batch["y"]
        preds = self(x).squeeze(-1)
        y = y.float()
        self.test_metrics.update(preds, y)

    def on_test_epoch_end(self) -> None:
        self._log_metrics(self.test_metrics, prefix="test")

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch["X"]
        return self(x).squeeze(-1)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _log_metrics(self, collection: MetricCollection, prefix: str) -> None:
        """Compute, log, and reset a MetricCollection."""
        metrics = collection.compute()
        self.log_dict(
            {f"{prefix}_{name}": value for name, value in metrics.items()},
            on_epoch=True,
            prog_bar=False,
        )
        collection.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
