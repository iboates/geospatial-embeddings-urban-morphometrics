"""
Registry + builder for SRAI embedders.  Each entry maps a config `name` key to
an (import_path, class_name) pair.  The `build_embedder` function handles the
different constructor signatures so experiment configs stay uniform.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────
EMBEDDER_REGISTRY: dict[str, tuple[str, str]] = {
    "Hex2Vec": ("srai.embedders", "Hex2VecEmbedder"),
    "GeoVex": ("srai.embedders", "GeoVexEmbedder"),
    "CountEmbedder": ("srai.embedders", "CountEmbedder"),
    "ContextualCountEmbedder": ("srai.embedders", "ContextualCountEmbedder"),
    "UrbanMorphometricsEmbedder": (
        "urban_morphometrics.embedding.embedders.urban_morphometrics_embedder",
        "UrbanMorphometricsEmbedder",
    ),
    "Hex2VecUrbanMorphometrics": (
        "urban_morphometrics.embedding.embedders.hex2vec_urban_morphometrics_embedder",
        "UrbanMorphometricsHex2VecEmbedder",
    ),
    # ← Add more embedders here
}

# Embedders that do NOT need a fit() call with OSM features
NO_FIT_EMBEDDERS = {"CountEmbedder", "UrbanMorphometricsEmbedder"}


def build_embedder(
    name: str,
    hidden_sizes: list[int],
    osm_filter: Any,
    morpho_filter: Any | None = None,
    neighbourhood: Any | None = None,
    neighbourhood_radius: int = 2,
) -> Any:
    """
    Construct an embedder by registry key.

    Args:
        name: Registry key (e.g. "Hex2Vec").
        hidden_sizes: Architecture sizes (ignored for embedders that don't use them).
        osm_filter: OSM feature filter (passed as `target_features` or
                    `expected_output_features` depending on embedder).
        morpho_filter: Urban Morphology Metrics filter
        neighbourhood: H3Neighbourhood (required for ContextualCountEmbedder).
        neighbourhood_radius: Radius for neighbourhood-based embedders.

    Returns:
        Instantiated (unfitted) embedder.
    """
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedder '{name}'. Available: {list(EMBEDDER_REGISTRY.keys())}"
        )

    module_path, class_name = EMBEDDER_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # ── Per-embedder construction ─────────────────────────────────────────────
    if name == "Hex2Vec":
        embedder = cls(hidden_sizes, expected_output_features=osm_filter)

    elif name == "GeoVex":
        embedder = cls(
            target_features=osm_filter,
            neighbourhood_radius=neighbourhood_radius,
        )

    elif name == "CountEmbedder":
        embedder = cls(expected_output_features=osm_filter)

    elif name == "ContextualCountEmbedder":
        if neighbourhood is None:
            raise ValueError("ContextualCountEmbedder requires a neighbourhood object.")
        embedder = cls(
            neighbourhood=neighbourhood,
            neighbourhood_distance=neighbourhood_radius,
            expected_output_features=osm_filter,
            concatenate_vectors=True,
            count_subcategories=True,
        )

    elif name in ("UrbanMorphometricsEmbedder", "Hex2VecUrbanMorphometrics"):
        if morpho_filter is None:
            raise ValueError(f"{name} requires a `morpho_filter` object")
        embedder = cls(
            expected_output_features=osm_filter,
            expected_morphology_features=morpho_filter,
        )

    else:
        # Generic fallback — try passing hidden_sizes if accepted
        try:
            embedder = cls(hidden_sizes)
        except TypeError:
            embedder = cls()

    logger.info("Built embedder '%s'", name)
    return embedder


def requires_fit(name: str) -> bool:
    """Return False for embedders that skip the fit step."""
    return name not in NO_FIT_EMBEDDERS
