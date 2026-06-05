import argparse
from pathlib import Path

import pandas as pd
import torch
from captum.attr import IntegratedGradients
from dotenv import load_dotenv
from srai.regionalizers import H3Regionalizer
from torch.utils.data import DataLoader

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
from urban_morphometrics.embedding.models.regression import RegressionBaseModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# 1. Load Lightning model
# ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path, model_cfg, embedding_size, train_cfg):
    model = RegressionBaseModel.load_from_checkpoint(
        checkpoint_path,
        embeddings_size=embedding_size,
        linear_sizes=model_cfg["linear_sizes"],
        dropout_p=model_cfg.get("dropout_p", 0.2),
        loss_name=train_cfg["loss"],
    )
    model.eval()
    model.to(DEVICE)
    return model


# ─────────────────────────────────────────────────────────────
# 2. Wrapper (ensures clean tensor output)
# ─────────────────────────────────────────────────────────────
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)  # must return tensor [B, 1] or [B]


# ─────────────────────────────────────────────────────────────
# 3. Get tensors from dataloader
# ─────────────────────────────────────────────────────────────
def get_tensor_batch(dataloader, n_samples=100):
    X_list = []

    for batch in dataloader:
        X = batch["X"]
        X_list.append(X)

        if sum(len(x) for x in X_list) >= n_samples:
            break

    X = torch.cat(X_list, dim=0)[:n_samples]
    return X.to(DEVICE)


# ─────────────────────────────────────────────────────────────
# 4. Run Captum (Integrated Gradients)
# ─────────────────────────────────────────────────────────────
def run_captum(model, train_loader, test_loader, feat_cols, output_dir):
    wrapped_model = WrappedModel(model)

    ig = IntegratedGradients(wrapped_model)

    # baseline = mean of training data (important!)
    background = get_tensor_batch(train_loader, n_samples=200)
    baseline = background.mean(dim=0, keepdim=True)

    # samples to explain
    test_samples = get_tensor_batch(test_loader, n_samples=100)

    # compute attributions
    attributions = ig.attribute(test_samples, baselines=baseline, n_steps=50)

    attributions = attributions.detach().cpu()

    # ── Save raw attributions ────────────────────────────────
    torch.save(attributions, Path(output_dir) / "captum_attributions.pt")

    # ── Global feature importance ────────────────────────────
    importance = attributions.abs().mean(dim=0).numpy()

    df = pd.DataFrame({"feature": feat_cols, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    df.to_csv(Path(output_dir) / "captum_feature_importance.csv", index=False)

    # ── Optional: simple bar plot ────────────────────────────
    import matplotlib.pyplot as plt

    top_k = 20
    df_top = df.head(top_k)

    plt.figure()
    plt.barh(df_top["feature"], df_top["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances (Captum IG)")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "captum_feature_importance.png")
    plt.close()

    print("Captum analysis saved.")


# ─────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────
def run_captum_pipeline(
    cfg, output_dir, train_loader, test_loader, embedding_size, feat_cols
):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    checkpoint_path = (
        Path(output_dir)
        / f"{cfg['sweep_name']}_{cfg['experiment_name']}_{cfg['dataset']['name']}_r{cfg['resolution']}_best_model.pt.ckpt"
    )

    model = load_model(checkpoint_path, model_cfg, embedding_size, train_cfg)

    run_captum(model, train_loader, test_loader, feat_cols, output_dir)


def run_captum_from_config(config_path: str):
    # reuse your existing pipeline up to dataloaders + metadata
    cfg = load_config(config_path)

    # ── replicate ONLY what we need from run() ──
    ds_cfg = cfg["dataset"]
    dataset = load_dataset(ds_cfg["name"], ds_cfg["version"])

    train_gdf, dev_gdf = dataset.train_test_split(
        test_size=ds_cfg["dev_size"], validation_split=True
    )
    _, test_gdf = dataset.load(version=ds_cfg["version"]).values()

    regionalizer = H3Regionalizer(resolution=cfg["resolution"])

    joined_train, regions_train = assign_h3_index(train_gdf, regionalizer)
    joined_dev, regions_dev = assign_h3_index(dev_gdf, regionalizer)
    joined_test, regions_test = assign_h3_index(test_gdf, regionalizer)

    full_regions = build_full_regions(
        regions_train, regionalizer, ds_cfg["name"], train_gdf
    )

    use_numerical = ds_cfg["use_numerical_columns"]
    numerical_cols = dataset.numerical_columns if use_numerical else []

    scaler = None
    if use_numerical:
        joined_train, scaler = fit_transform_scaler(joined_train, numerical_cols)
        joined_dev = transform_scaler(joined_dev, numerical_cols, scaler)
        joined_test = transform_scaler(joined_test, numerical_cols, scaler)

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

    # ── embeddings (reuse your pipeline) ──
    emb_cfg = cfg["embedder"]
    osm_filter = get_osm_filter(cfg["osm_filter"])
    morpho_filter = get_morpho_filter(cfg["morpho_filter"])

    embedder = build_embedder(
        name=emb_cfg["name"],
        hidden_sizes=emb_cfg.get("hidden_sizes", []),
        osm_filter=osm_filter,
        morpho_filter=morpho_filter,
        neighbourhood_radius=cfg["neighbourhood_radius"],
    )

    emb_train, emb_dev, emb_test = run_embedding_pipeline(
        embedder=embedder,
        embedder_name=emb_cfg["name"],
        exp_name="captum_eval",
        regions_train=regions_train,
        regions_dev=regions_dev,
        regions_test=regions_test,
        full_regions=full_regions,
        osm_filter=osm_filter,
        neighbourhood_radius=cfg["neighbourhood_radius"],
        fit_kwargs=emb_cfg.get("fit_kwargs", {}),
        morpho_cfg=cfg.get("morphometrics", {}),
    )

    merged_train, feat_cols = merge_embeddings_with_targets(
        emb_train, aggregated_train, dataset.target
    )
    merged_test, _ = merge_embeddings_with_targets(
        emb_test, aggregated_test, dataset.target
    )

    # ── dataloaders ──
    batch_size = cfg["training"]["batch_size"]

    hf_train = build_hf_dataset(merged_train, feat_cols, dataset.target)
    hf_test = build_hf_dataset(merged_test, feat_cols, dataset.target)

    train_loader = DataLoader(hf_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(hf_test, batch_size=batch_size, shuffle=False)

    embedding_size = hf_train["X"].shape[1]

    # ── reconstruct output_dir EXACTLY like training ──
    exp_name = (
        f"{cfg.get('sweep_name', '')}_"
        f"{cfg.get('experiment_name', '')}_"
        f"{ds_cfg['name']}_r{cfg['resolution']}"
    )

    output_dir = Path(cfg["output_dir"]) / exp_name

    # ── run captum ──
    run_captum_pipeline(
        cfg,
        output_dir,
        train_loader,
        test_loader,
        embedding_size,
        feat_cols,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Captum attribution on a trained regression model."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    run_captum_from_config(args.config)


if __name__ == "__main__":
    load_dotenv()
    main()
