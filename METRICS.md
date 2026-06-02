# Metrics Inventory

This document catalogues every metric computed by the pipeline, grouped by category. Each entry explains what the metric measures and whether it requires **neighbourhood context** — data from features beyond the focal cell boundary — to produce correct results.

**Neighbourhood context** means the metric builds a spatial graph (tessellation, KNN, Delaunay, contiguity, or street network) or performs a spatial lookup (nearest street) whose results are distorted if the input is clipped to the cell boundary. These metrics receive buildings/highways from the focal cell *plus adjacent H3 cells* so that edge buildings have complete spatial relationships.

---

## Dimension Metrics

| Metric | momepy function | Neighbourhood | What it measures |
|--------|----------------|:---:|-----------------|
| `courtyard_area` | *(custom)* | **Yes** | Area of interior courtyards. Focal and neighbourhood buildings are dissolved into unified structures; each interior ring (hole) that intersects a focal building is extracted as an individual courtyard polygon. Statistics are computed over individual courtyard areas, plus a count. |
| `floor_area` | *(custom)* | No | Footprint area multiplied by number of floors. Floor count comes from OSM `building:levels` tag, or `height / 3m` as fallback. |
| `longest_axis_length` | `momepy.longest_axis_length` | No | Diameter of the minimum bounding circle of each building footprint. Measures the longest dimension of the building. |
| `perimeter_wall_individual` | *(custom)* | No | Outer perimeter of each individual building footprint (`geometry.length` in equidistant CRS). Values in metres. |
| `perimeter_wall_joined` | *(custom)* | No | Perimeter of joined (touching) structures. Buildings are dissolved via `unary_union`; one perimeter value is computed per dissolved structure (not per building), avoiding duplication in the statistics when multiple buildings share a structure. Values in metres. |
| `volume` | *(custom)* | No | Footprint area multiplied by building height. Height comes from OSM `height` tag, or `building:levels * 3m`, or defaults to 6m. |

---

## Shape Metrics

Most shape metrics are self-contained per-building calculations with no neighbourhood dependency. Exception: `courtyard_index` requires neighbourhood context.

| Metric | momepy function | What it measures |
|--------|----------------|-----------------|
| `circular_compactness` | `momepy.circular_compactness` | Ratio of building area to the area of its minimum bounding circle. Values near 1 indicate circular footprints; lower values indicate irregular shapes. |
| `square_compactness` | `momepy.square_compactness` | `(4 * sqrt(area) / perimeter)^2`. Measures how efficiently the perimeter encloses area. A perfect square scores 1. |
| `convexity` | `momepy.convexity` | Ratio of building area to its convex hull area. Values near 1 mean the footprint has no concavities; lower values indicate L-shapes, courtyards, or other concave forms. |
| `courtyard_index` | `momepy.courtyard_index` | **Neighbourhood** Ratio of courtyard (interior hole) area to total footprint area. Focal and neighbourhood buildings are dissolved together so boundary-spanning courtyards are captured correctly. Only dissolved structures intersecting the focal cell that contain at least one interior ring are included. Cells with no courtyards produce all-NaN aggregated statistics. |
| `rectangularity` | `momepy.rectangularity` | Ratio of building area to its minimum rotated bounding rectangle. Values near 1 indicate rectangular footprints. |
| `shape_index` | `momepy.shape_index` | `sqrt(area / pi) / (0.5 * longest_axis)`. Measures how close the shape is to a circle. A perfect circle scores 1. |
| `corners` | `momepy.corners` | Count of vertices where the interior angle deviates significantly from 180°. Captures footprint complexity. |
| `squareness` | `momepy.squareness` | Mean deviation of corner angles from 90°. Low values indicate buildings with predominantly right-angle corners (typical of modern construction). |
| `equivalent_rectangular_index` | `momepy.equivalent_rectangular_index` | Ratio comparing the building's area and perimeter to an equivalent rectangle. High ERI → rectangular (row houses, slabs); low ERI → complex footprints (L-shapes, courtyards). |
| `elongation` | `momepy.elongation` | Ratio of the shorter to the longer side of the minimum bounding rectangle. Values near 1 → compact/square; low values → elongated (row houses, strips). |
| `facade_ratio` | `momepy.facade_ratio` | Ratio of building area to perimeter. Computed twice: once per raw building, once per dissolved structure (adjacent buildings merged). |
| `fractal_dimension` | `momepy.fractal_dimension` | `2 * log(perimeter/4) / log(area)`. ~1.0 = very simple (square); 1.05–1.15 = moderately articulated; >1.20 = complex/fragmented. Computed per raw building and per dissolved structure. |
| `form_factor` | `momepy.form_factor` | `surface / volume^(2/3)`. A 3D compactness measure using building height. Low values indicate compact blocks; high values indicate thin or tall buildings. |
| `compactness_weighted_axis` | `momepy.compactness_weighted_axis` | Combines longest axis length with compactness. Measures how efficiently a polygon fills space relative to its principal axes. Computed per raw building and per dissolved structure. |
| `centroid_corner_distance` | `momepy.centroid_corner_distance` | Mean and standard deviation of distances from the building centroid to each vertex. Captures both size and irregularity of the footprint. Computed per raw building and per dissolved structure. |

---

## Distribution Metrics

These metrics describe the spatial arrangement, orientation, and relationships *between* buildings (or between buildings and streets). Many require neighbourhood context because they build spatial graphs (tessellation, KNN, Delaunay triangulation, contiguity) that are distorted when clipped at a cell boundary.

| Metric | momepy function | Neighbourhood | What it measures |
|--------|----------------|:---:|-----------------|
| `orientation` | `momepy.orientation` | No | Deviation of each building's longest axis from cardinal directions (0°–45° range). Low mean orientation → grid-aligned; high values → organic layout. |
| `shared_walls` | `momepy.shared_walls` | **Yes** | Length of wall shared between adjacent (touching) buildings. Also computes a ratio variant: shared wall length as a fraction of each building's total perimeter. High values indicate terraced/attached housing. |
| `alignment` | `momepy.alignment` | **Yes** | Consistency of orientation among neighbouring buildings. Uses a KNN graph to define neighbours (`knn_k` parameter, default 15), then measures how similarly oriented each building is relative to its neighbours. Low values → uniform grid; high values → heterogeneous orientations. |
| `neighbor_distance` | `momepy.neighbor_distance` | **Yes** | Mean distance from each building to its Delaunay-triangulation neighbours. Captures typical spacing between nearby buildings. |
| `mean_interbuilding_distance` | `momepy.mean_interbuilding_distance` | **Yes** | Mean distance between a building and all buildings within its Delaunay+KNN neighbourhood (`knn_k` parameter, default 15). A broader spacing measure than `neighbor_distance`. |
| `building_adjacency` | `momepy.building_adjacency` | **Yes** | Ratio of buildings that share a wall (rook contiguity) to the number of KNN neighbours (`knn_k` parameter, default 15). High values indicate predominantly attached/terraced buildings. |
| `neighbors` | `momepy.neighbors` | **Yes** | Number of neighbours for each building's morphological tessellation cell, measured by queen contiguity on the tessellation. The tessellation is a Voronoi-like partitioning of space where each building "owns" the area closest to it (clipped by a buffered convex hull). Queen contiguity counts tessellation cells that share any boundary point with the focal cell — i.e. all buildings whose territory touches the focal building's territory. High values → densely surrounded building; low values → isolated building with few close neighbours. Tessellation parameters control the shape of the cells and are configurable via `--metric-config`: `tessellation_buffer` (default 5), `tessellation_min_buffer` (default 0), `tessellation_max_buffer` (default 10), `tessellation_shrink` (default 0.4m), `tessellation_segment` (default 0.5m). |
| `cell_alignment` | `momepy.cell_alignment` | **Yes** | Deviation between each building's orientation and the orientation of its tessellation cell. Measures how well aligned buildings are with the local urban fabric (Voronoi partitioning). |
| `street_alignment` | `momepy.street_alignment` | **Yes** | Deviation between each building's orientation and the orientation of its nearest street. Uses `momepy.get_nearest_street` to match buildings to streets (`street_alignment_max_distance` parameter, default 500m; buildings with no street within this radius get NaN), then compares orientations. Low values → buildings parallel/perpendicular to streets. |

---

## Intensity Metrics

| Metric | momepy function | Neighbourhood | What it measures |
|--------|----------------|:---:|-----------------|
| `courtyards` | *(custom)* | **Yes** | Total number of courtyards in the focal cell. All buildings (focal + neighbourhood) are dissolved into superstructures; interior rings of superstructures that intersect any focal building are counted. Returns a single `courtyards_count` value per cell. |

---

## Street Relationship Metrics

| Metric | momepy function | Neighbourhood | What it measures |
|--------|----------------|:---:|-----------------|
| `street_profile` | `momepy.street_profile` | **Yes** | Street width, openness, and height-to-width ratio measured by casting perpendicular ticks from street centrelines and detecting flanking buildings. Requires neighbourhood buildings so ticks at cell edges can reach buildings outside the focal cell. |
| `nearest_street_distance` | *(custom)* | **Yes** | Minimum distance from each building centroid to the nearest street segment. Requires neighbourhood highways so buildings near cell edges find their true nearest street. |

---

## Street Connectivity Metrics

All connectivity metrics require neighbourhood context because the street network graph is severely distorted when truncated at a cell boundary — centrality measures, cycle counts, and node degrees all depend on the wider network topology. Each metric below is computed separately for the **vehicle** network (directed graph respecting one-way streets) and the **pedestrian** network (undirected graph).

**Graph cleaning:** Before building either graph, degree-2 interstitial nodes (passthrough points that are not true intersections) are dissolved using `remove_interstitial_nodes_preserving_oneway` in `metrics/_utils.py`. A node is only removed when exactly 2 edges meet *and* both share the same `oneway` value — nodes at a oneway/bidirectional boundary (e.g. where a one-way street meets a two-way street) are preserved even if they are degree 2, preventing corruption of the directed graph.

### Per-Node Metrics (aggregated: mean, median, std, deciles)

| Metric | momepy function | What it measures |
|--------|----------------|-----------------|
| `degree` | `momepy.node_degree` | Number of edges meeting at each intersection. 1 = dead end; 3 = T-junction; 4 = crossroads. |
| `meshedness` | `momepy.meshedness` | Local ratio of independent cycles to the maximum possible within a subgraph around each node (`network_subgraph_radius` hops, default 5). High values → grid-like network with many alternative routes. |
| `mean_node_dist` | `momepy.mean_node_dist` | Mean distance between a node and its direct graph neighbours. Captures typical block length / intersection spacing. |
| `mean_node_degree` | `momepy.mean_node_degree` | Mean degree within a local subgraph around each node (`network_subgraph_radius` hops, default 5). Smoothed version of raw degree. |
| `gamma` | `momepy.gamma` | Local ratio of observed edges to maximum possible edges within a subgraph around each node (`network_subgraph_radius` hops, default 5). Measures local connectivity completeness. |
| `edge_node_ratio` | `momepy.edge_node_ratio` | Local ratio of edges to nodes within a subgraph around each node (`network_subgraph_radius` hops, default 5). Higher values → more connections per intersection. |
| `cyclomatic` | `momepy.cyclomatic` | Number of independent cycles in a local subgraph around each node (`network_subgraph_radius` hops, default 5). More cycles → more route alternatives. |
| `clustering` | `momepy.clustering` | Clustering coefficient — fraction of a node's neighbours that are also connected to each other. High values → tightly interconnected local network. |
| `closeness_centrality` | `momepy.closeness_centrality` | Inverse of the mean shortest-path distance to all other nodes. High values → nodes close to the centre of the network. |
| `betweenness_centrality` | `momepy.betweenness_centrality` | Fraction of all shortest paths that pass through each node. High values → key through-traffic intersections. |
| `straightness_centrality` | `momepy.straightness_centrality` | Ratio of Euclidean distance to network distance for paths through each node. High values → direct, straight connections. |

### Graph-Level Metrics (single value per cell)

| Metric | momepy function | What it measures |
|--------|----------------|-----------------|
| `meshedness_global` | `momepy.meshedness` (radius=None) | Global meshedness of the entire cell's network. |
| `gamma_global` | `momepy.gamma` (radius=None) | Global gamma index (edge-to-max-edge ratio) for the entire cell's network. |
| `cds_length_total` | `momepy.cds_length` | Total length of cul-de-sac (dead-end) segments. |

### Proportion Metrics (single value per cell)

| Metric | momepy function | What it measures |
|--------|----------------|-----------------|
| `proportion_three` | `momepy.proportion` | Fraction of intersections that are 3-way (T-junctions). |
| `proportion_four` | `momepy.proportion` | Fraction of intersections that are 4-way (crossroads). |
| `proportion_dead` | `momepy.proportion` | Fraction of nodes that are dead ends. |

---

## Summary: Which Metrics Need Neighbourhood Context?

**Self-contained** (focal-cell data only — 20 metrics):
`floor_area`, `longest_axis_length`, `perimeter_wall_individual`, `perimeter_wall_joined`, `volume`, `circular_compactness`, `square_compactness`, `convexity`, `rectangularity`, `shape_index`, `corners`, `squareness`, `equivalent_rectangular_index`, `elongation`, `facade_ratio`, `fractal_dimension`, `form_factor`, `compactness_weighted_axis`, `centroid_corner_distance`, `orientation`

**Require neighbourhood context** (data from beyond the cell boundary — 12 building/street metrics + 34 connectivity metrics):
`courtyard_area`, `courtyard_index`, `courtyards`, `shared_walls`, `alignment`, `neighbor_distance`, `mean_interbuilding_distance`, `building_adjacency`, `neighbors`, `cell_alignment`, `street_alignment`, `street_profile`, `nearest_street_distance`, and all `*_vehicle` / `*_pedestrian` connectivity metrics
