"""
Urban Morphometrics Hex2Vec Embedder.

This module contains an embedder that combines Hex2Vec region embeddings
with pre-calculated morphological metrics.

References:
    [1] https://dl.acm.org/doi/10.1145/3486635.3491076
"""

from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from srai.constants import GEOMETRY_COLUMN
from srai.embedders.hex2vec.embedder import Hex2VecEmbedder
from srai.embedders.hex2vec.model import Hex2VecModel
from srai.embedders.hex2vec.neighbour_dataset import NeighbourDataset
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter
from srai.neighbourhoods import Neighbourhood

T = TypeVar("T")


class UrbanMorphometricsHex2VecEmbedder(Hex2VecEmbedder):
    """
    Hex2Vec Embedder that incorporates Urban Morphological metrics into the learned embedding.

    Unlike a simple post-hoc concatenation, morphological features (e.g. building footprint
    ratios, street network orientation entropy) are joined with the OSM count features
    BEFORE the encoder sees any data. This means the Hex2Vec model is trained on — and
    produces embeddings from — the full combined feature space, so morphological structure
    is captured in the latent representation itself rather than appended afterwards.
    """

    def __init__(
        self,
        encoder_sizes: Optional[list[int]] = None,
        expected_output_features: Optional[
            Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter]
        ] = None,
        expected_morphology_features: Optional[list[str]] = None,
        count_subcategories: bool = True,
    ) -> None:
        """
        Initialize UrbanMorphometricsHex2VecEmbedder.

        Args:
            encoder_sizes (List[int], optional): Sizes of the Hex2Vec encoder layers.
                The input size is inferred from the combined (counts + morpho) feature
                width at fit time. Defaults to [150, 75, 50].
            expected_output_features
                (Union[List[str], OsmTagsFilter, GroupedOsmTagsFilter], optional):
                OSM features expected in the count embedding input. Defaults to None.
            expected_morphology_features (List[str], optional): Column name prefixes
                used to select morphological metric columns from morpho_features_gdf.
                Matching uses str.startswith, so both exact names and shared prefixes
                (e.g. "lcd_", "stc_") are supported. When None, all columns are kept.
                Defaults to None.
            count_subcategories (bool, optional): Whether to count all OSM subcategories
                individually or only at the top-level tag. Defaults to True.
        """
        super().__init__(
            encoder_sizes=encoder_sizes,
            expected_output_features=expected_output_features,
            count_subcategories=count_subcategories,
        )
        self.expected_morphology_features = (
            tuple(expected_morphology_features)
            if expected_morphology_features is not None
            else None
        )

    def _get_combined_features(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        morpho_features_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Build the combined (OSM counts + morphological metrics) feature matrix.

        This is the single source of truth for both fit() and transform(), ensuring
        the model always sees features constructed identically at training and inference time.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and OSM values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            morpho_features_gdf (gpd.GeoDataFrame): Pre-aggregated morphological features
                indexed by the region ID (e.g. H3 index).

        Returns:
            pd.DataFrame: Float32 combined feature matrix aligned on the region index.
        """
        # OSM count features — already float32 via the parent helper
        if self.expected_output_features is not None:
            counts_df = self._get_raw_counts(regions_gdf, features_gdf, joint_gdf)
        else:
            counts_df = pd.DataFrame(regions_gdf).drop(columns=[GEOMETRY_COLUMN])

        # Morphological features
        morpho_df = pd.DataFrame(morpho_features_gdf)

        if GEOMETRY_COLUMN in morpho_df.columns:
            morpho_df = morpho_df.drop(columns=[GEOMETRY_COLUMN])

        if self.expected_morphology_features is not None:
            morpho_df = morpho_df[
                morpho_df.columns[
                    morpho_df.columns.str.startswith(self.expected_morphology_features)
                ]
            ]

        # Left-join so every region in counts_df has a row; fill gaps with 0
        combined_df = counts_df.join(morpho_df, how="left").fillna(0)

        return combined_df.astype(np.float32)

    def fit(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        neighbourhood: Neighbourhood[T],
        morpho_features_gdf: gpd.GeoDataFrame,
        negative_sample_k_distance: int = 2,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Fit the Hex2Vec model on the combined OSM count + morphological feature matrix.

        The model input width is determined by the full combined feature space, so the
        encoder captures both land-use composition and morphological structure in the
        learned embeddings.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and OSM values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            neighbourhood (Neighbourhood[T]): Neighbourhood initialised with the same regions.
            morpho_features_gdf (gpd.GeoDataFrame): Pre-aggregated morphological features
                indexed by region ID.
            negative_sample_k_distance (int, optional): Negative sample distance threshold.
                Defaults to 2.
            batch_size (int, optional): Training batch size. Defaults to 32.
            learning_rate (float, optional): Optimiser learning rate. Defaults to 0.001.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Extra kwargs forwarded
                to pytorch_lightning.Trainer. Defaults to None.
        """
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader

        trainer_kwargs = self._prepare_trainer_kwargs(trainer_kwargs)

        combined_df = self._get_combined_features(
            regions_gdf, features_gdf, joint_gdf, morpho_features_gdf
        )

        # # Mirror the parent's expected_output_features freeze, but across the full
        # # combined column set so save/load can reconstruct the input layer correctly.
        # if self.expected_output_features is None:
        #     self.expected_output_features = pd.Series(combined_df.columns)

        num_features = len(combined_df.columns)
        self._model = Hex2VecModel(
            layer_sizes=[num_features, *self._encoder_sizes],
            learning_rate=learning_rate,
        )

        dataset = NeighbourDataset(
            combined_df, neighbourhood, negative_sample_k_distance
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self._model, dataloader)
        self._is_fitted = True

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        morpho_features_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Produce region embeddings from the combined OSM count + morphological feature matrix.

        The same feature construction used in fit() is applied here, so the encoder
        receives inputs with the identical structure it was trained on.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and OSM values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            morpho_features_gdf (gpd.GeoDataFrame): Pre-aggregated morphological features
                indexed by region ID.

        Returns:
            pd.DataFrame: Hex2Vec embeddings for each region.

        Raises:
            ModelNotFitException: If fit() has not been called yet.
        """
        import torch

        self._check_is_fitted()

        combined_df = self._get_combined_features(
            regions_gdf, features_gdf, joint_gdf, morpho_features_gdf
        )

        combined_tensor = torch.from_numpy(combined_df.values)
        embeddings = self._model(combined_tensor).detach().numpy()  # type: ignore

        return pd.DataFrame(embeddings, index=combined_df.index)

    def fit_transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        neighbourhood: Neighbourhood[T],
        morpho_features_gdf: gpd.GeoDataFrame,
        negative_sample_k_distance: int = 2,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        trainer_kwargs: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fit the model and return embeddings in one step.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries and OSM values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            neighbourhood (Neighbourhood[T]): Neighbourhood initialised with the same regions.
            morpho_features_gdf (gpd.GeoDataFrame): Pre-aggregated morphological features
                indexed by region ID.
            negative_sample_k_distance (int, optional): Defaults to 2.
            batch_size (int, optional): Defaults to 32.
            learning_rate (float, optional): Defaults to 0.001.
            trainer_kwargs (Optional[Dict[str, Any]], optional): Defaults to None.

        Returns:
            pd.DataFrame: Hex2Vec embeddings for each region.
        """
        self.fit(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            joint_gdf=joint_gdf,
            neighbourhood=neighbourhood,
            morpho_features_gdf=morpho_features_gdf,
            negative_sample_k_distance=negative_sample_k_distance,
            batch_size=batch_size,
            learning_rate=learning_rate,
            trainer_kwargs=trainer_kwargs,
        )
        return self.transform(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            joint_gdf=joint_gdf,
            morpho_features_gdf=morpho_features_gdf,
        )

    def save(self, path: Union[Path, str]) -> None:
        """
        Save the fitted model and embedder configuration to a directory.

        Args:
            path (Union[Path, str]): Target directory path.
        """
        embedder_config = {
            "encoder_sizes": self._encoder_sizes,
            "expected_output_features": (
                self.expected_output_features.tolist()
                if self.expected_output_features is not None
                else None
            ),
            "expected_morphology_features": (
                list(self.expected_morphology_features)
                if self.expected_morphology_features is not None
                else None
            ),
        }
        self._save(path, embedder_config)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "UrbanMorphometricsHex2VecEmbedder":
        """
        Load a saved embedder from a directory.

        Args:
            path (Union[Path, str]): Directory produced by save().

        Returns:
            UrbanMorphometricsHex2VecEmbedder: The restored embedder.
        """
        return cls._load(path, Hex2VecModel)
