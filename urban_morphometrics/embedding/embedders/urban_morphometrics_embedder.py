"""
Urban Morphometrics Embedder.

This module contains an embedder that combines count-based OSM features
with pre-calculated morphological metrics.
"""

from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from srai.constants import GEOMETRY_COLUMN
from srai.embedders import CountEmbedder
from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter, OsmTagsFilter


class UrbanMorphometricsEmbedder(CountEmbedder):
    """
    Embedder that combines OSM feature counts with pre-aggregated morphological metrics.

    Inherits from CountEmbedder to calculate occurrences of OSM features, and then
    concatenates externally provided morphological features (like building footprints,
    street orientations, etc.) indexed by the region ID.
    """

    def __init__(
        self,
        expected_output_features: Optional[
            Union[list[str], OsmTagsFilter, GroupedOsmTagsFilter]
        ] = None,
        expected_morphology_features: Optional[list[str]] = None,
        count_subcategories: bool = True,
    ) -> None:
        """
        Init UrbanMorphometricsEmbedder.

        Args:
            expected_output_features: The OSM features expected to be found in the resulting embedding.
            expected_morphology_features: The Urban Morphology Metrics to be found in the resulting embedding.
            count_subcategories: Whether to count all OSM subcategories individually.
        """
        # Pass initialization logic entirely to the parent CountEmbedder
        super().__init__(
            expected_output_features=expected_output_features,
            count_subcategories=count_subcategories,
        )

        self.expected_morphology_features = tuple(expected_morphology_features)

    def transform(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
        morpho_features_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Embed a given GeoDataFrame with both counts and morpho metrics.

        Creates region embeddings by counting the frequencies of each OSM feature value
        and concatenating them with pre-calculated morphological metrics.

        Args:
            regions_gdf (gpd.GeoDataFrame): Region indexes and geometries.
            features_gdf (gpd.GeoDataFrame): Feature indexes, geometries, and OSM values.
            joint_gdf (gpd.GeoDataFrame): Joiner result with region-feature multi-index.
            morpho_features_gdf (gpd.GeoDataFrame): Pre-aggregated morphological features,
                indexed by the region ID (e.g., H3 index).

        Returns:
            pd.DataFrame: Combined embedding for each region.
        """
        # 1. Generate the standard count embeddings using the parent class method
        if self.expected_output_features is not None:
            count_embeddings_df = super().transform(
                regions_gdf=regions_gdf,
                features_gdf=features_gdf,
                joint_gdf=joint_gdf,
            )
        else:
            count_embeddings_df = pd.DataFrame(regions_gdf).drop(
                columns=[GEOMETRY_COLUMN]
            )

        # 2. Process the morphological features GeoDataFrame
        morpho_df = pd.DataFrame(morpho_features_gdf)

        # Embeddings should only contain numerical features, so we drop the geometry column
        if GEOMETRY_COLUMN in morpho_df.columns:
            morpho_df = morpho_df.drop(columns=[GEOMETRY_COLUMN])

        morpho_df = morpho_df[
            morpho_df.columns[
                morpho_df.columns.str.startswith(self.expected_morphology_features)
            ]
        ]

        # 3. Concatenate the dataframes
        # Since both dataframes use the region ID as the index (e.g., REGIONS_INDEX),
        # a left join perfectly aligns the new features to the existing count embeddings.
        combined_embeddings_df = count_embeddings_df.join(morpho_df, how="left")

        # 4. Fill missing values
        # If any regions lacked morphological data, fill those NaN values with 0
        # to ensure the final embedding tensor remains valid for ML downstream tasks.
        combined_embeddings_df = combined_embeddings_df.fillna(0)

        return combined_embeddings_df
