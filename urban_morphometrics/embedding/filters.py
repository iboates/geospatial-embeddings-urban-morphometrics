"""Filter definitions for urban morphometrics embedding pipeline.

Use metric prefixes with startswith() to match all variations (mean, median, std, quantiles).
Example: col.startswith("floor_area") matches floor_area_mean, floor_area_std, floor_area_q50, etc.
"""

EMPTY_FILTER = None

ALL_FILTER = {
    # Dimension Metrics - building size and volume
    "courtyard_area": ["courtyard_area"],
    "floor_area": ["floor_area"],
    "longest_axis_length": ["longest_axis_length"],
    "perimeter_wall": ["perimeter_wall_individual", "perimeter_wall_joined"],
    # "volume": ["volume"],
    # Shape Metrics - building footprint geometry
    "centroid_corner_distance": [
        "ccd_mean",
        "ccd_std",
        "ccd_joined_mean",
        "ccd_joined_std",
    ],
    "circular_compactness": ["circular_compactness", "circular_compactness_joined"],
    "compactness_weighted_axis": [
        "compactness_weighted_axis",
        "compactness_weighted_axis_joined",
    ],
    "convexity": ["convexity", "convexity_joined"],
    "corners": ["corners", "corners_joined"],
    "courtyard_index": ["courtyard_index"],
    "elongation": ["elongation", "elongation_joined"],
    "equivalent_rectangular_index": [
        "equivalent_rectangular_index",
        "equivalent_rectangular_index_joined",
    ],
    "facade_ratio": ["facade_ratio", "facade_ratio_joined"],
    "form_factor": ["form_factor"],
    "fractal_dimension": ["fractal_dimension", "fractal_dimension_joined"],
    "rectangularity": ["rectangularity", "rectangularity_joined"],
    "shape_index": ["shape_index", "shape_index_joined"],
    "square_compactness": ["square_compactness", "square_compactness_joined"],
    "squareness": ["squareness", "squareness_joined"],
    # Distribution Metrics - building spatial arrangement
    "alignment": ["alignment"],
    "building_adjacency": ["building_adjacency"],
    "cell_alignment": ["cell_alignment"],
    "mean_interbuilding_distance": ["mean_interbuilding_distance"],
    "neighbor_distance": ["neighbor_distance"],
    "neighbors": ["neighbors"],
    "orientation": ["orientation"],
    "shared_walls": ["shared_walls", "shared_walls_ratio"],
    "street_alignment": ["street_alignment"],
    # Intensity Metrics - number of features
    "courtyards": ["courtyards_count"],
    # Street Relationship Metrics
    "nearest_street_distance": ["nearest_street_distance"],
    "street_profile": [
        "street_profile_width",
        "street_profile_openness",
        "street_profile_hw_ratio",
    ],
    # Street Connectivity - Per-Node Metrics (vehicle and pedestrian networks)
    "betweenness_centrality": [
        "betweenness_centrality_vehicle",
        "betweenness_centrality_pedestrian",
    ],
    "closeness_centrality": [
        "closeness_centrality_vehicle",
        "closeness_centrality_pedestrian",
    ],
    "clustering": ["clustering_vehicle", "clustering_pedestrian"],
    "cyclomatic": ["cyclomatic_vehicle", "cyclomatic_pedestrian"],
    "degree": ["degree_vehicle", "degree_pedestrian"],
    "edge_node_ratio": ["edge_node_ratio_vehicle", "edge_node_ratio_pedestrian"],
    "gamma": ["gamma_vehicle", "gamma_pedestrian"],
    "mean_node_degree": ["mean_node_degree_vehicle", "mean_node_degree_pedestrian"],
    "mean_node_dist": ["mean_node_dist_vehicle", "mean_node_dist_pedestrian"],
    "meshedness": ["meshedness_vehicle", "meshedness_pedestrian"],
    "straightness_centrality": [
        "straightness_centrality_vehicle",
        "straightness_centrality_pedestrian",
    ],
    # Street Connectivity - Global Metrics (single values per network type)
    "cds_length_total": ["cds_length_total_vehicle", "cds_length_total_pedestrian"],
    "gamma_global": ["gamma_global_vehicle", "gamma_global_pedestrian"],
    "meshedness_global": ["meshedness_global_vehicle", "meshedness_global_pedestrian"],
    "proportion": [
        "proportion_three_vehicle",
        "proportion_four_vehicle",
        "proportion_dead_vehicle",
        "proportion_three_pedestrian",
        "proportion_four_pedestrian",
        "proportion_dead_pedestrian",
    ],
    # Land Cover Metrics - proportional coverage
    # "landuse_cover": [
    #     "landuse_farmland_proportion",
    #     "landuse_residential_proportion",
    #     "landuse_grass_proportion",
    #     "landuse_forest_proportion",
    #     "landuse_meadow_proportion",
    #     "landuse_orchard_proportion",
    #     "landuse_farmyard_proportion",
    #     "landuse_industrial_proportion",
    #     "landuse_vineyard_proportion",
    #     "landuse_cemetery_proportion",
    #     "landuse_commercial_proportion",
    # ],
    # "pedestrian_area_cover": ["pedestrian_area_proportion"],
    # "water_cover": ["water_proportion"],
}
