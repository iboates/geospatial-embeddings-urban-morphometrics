Make a plan to implement such a pipeline:

Use Fire as a CLI.

It is entered by a single function "compute_urban_morphometrics" which takes the following as an input:

* A study_area_gdf geodataframe in WGS84 of polygons with an attribute "region_id" as its primary identifier
* A URL to a .pbf OSM dump file
* A "run_name" indicating the name of the pipeline run
* A "neighbourhood_distance" buffer value
* A number of quantiles value
* An output folder path into which to write its outputs
* A list of strings indicating which metrics to compute
* An equal area, equidistant and conformal CRS
* A debug parameter indicating to dump intermediate layer results to the output folder

Make sure that the output folder is structured cleanly so it is clear where each type of output went

## The process

### 1. Loading OSM data

We are going to use the momepy library to compute various urban morphology metrics on the buildings and streets of each region.

First, use QuackOSM's `convert_pbf_to_geodataframe` method to retrieve the following for the entire study area, i.e. the union of all the cells in the study_area_gdf:

* A buildings_gdf, filter `{"building": True, "building:levels": True, "height": True}`
* A highways gdf, filter `{"highway": True, "oneway": True}`

QuackOSM should cache these resuls automatically. Use keep_all_tags=False.

### 2. The CellContext

Create a class called "CellContext" which will lazily-compute the following:

1. Buildings projected to equal area, equidistant and conformal CRSs

2. Buildings should have their height value estimated using the following criteria:#

    # Resolve building height: prefer pre-parsed 'height' column, then
    # fall back to building_levels * 3m, then default 6m (≈2 storeys).
    if "height" not in buildings.columns:
        # Legacy path: try inferring from tags dict
        if "tags" in buildings.columns:
            buildings["height"] = buildings["tags"].apply(_infer_height_from_tags)
        else:
            buildings["height"] = np.nan

    # Where height is still missing, derive from building_levels
    if "building_levels" in buildings.columns:
        missing_height = buildings["height"].isna()
        has_levels = buildings["building_levels"].notna()
        buildings.loc[missing_height & has_levels, "height"] = (
            buildings.loc[missing_height & has_levels, "building_levels"] * 3.0
        )

    buildings["height"] = buildings["height"].fillna(6.0)


3. Highways projected to equal area, equidistant and conformal CRSs, and filtered to have the following highway tag values (call it "vehicle_highways"): 

    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "road", "track"

    Vehicle highways should also have their oneway status set using the following logic:

    
def _parse_oneway(tags, highway_val) -> bool:
    """Parse OSM oneway tag. Returns True if one-way only, False if bidirectional.

    OSM oneway=yes/true/1 -> one-way. oneway=no/false/0 or absent -> two-way.
    highway=motorway and junction=roundabout default to one-way.
    """
    if tags is None or not isinstance(tags, dict):
        return _default_oneway(highway_val, None)
    oneway = tags.get("oneway")
    junction = tags.get("junction")
    if oneway is None:
        return _default_oneway(highway_val, junction)
    val = str(oneway).strip().lower()
    if val in ("yes", "true", "1"):
        return True
    if val in ("no", "false", "0"):
        return False
    if val == "-1":
        return True  # one-way reverse; geometry direction unchanged for now
    return _default_oneway(highway_val, junction)


def _default_oneway(highway_val, junction) -> bool:
    """Default oneway when tag absent: motorway and roundabout are one-way."""
    if junction and str(junction).lower() == "roundabout":
        return True
    if highway_val and str(highway_val).lower() in ("motorway", "motorway_link"):
        return True
    return False

4. Highways projected to equal area, equidistant and conformal CRSs, and filtered to have the following highway tag values (call it "pedestrian_highways"): 

    "unclassified", "residential",
    "service", "living_street", "road", "track",
    "footway", "pedestrian", "path", "steps", "corridor", "bridleway",

5. Buildings that are within neighbourhood_distance of the cell but do not intersect it
6. Highways that are within neighbourhood_distance of the cell but do not intersect it

The CellContext should cache its content based on the run name and load it if it already exists

### 3. Metrics computation

Each polygon (cell) in study_area_gdf must have the metrics computed about it using the metrics detailed in METRICS.md.

Neighbourhood metrics should combine the features in the focal cell with its neighbourhood values to ensure that all features that are actually inside the focal cell are the ones whose computed values are contributing to the metrics, but they are not suffering from edge conditions by considering some features outside of their parent cell boundary.

Most of the metrics rely on computing them on a feature-level. In those cases, the following values about the metric should be computed:

* Mean
* Median
* Standard Deviation
* Quantiles, as specified by an input parameter indicating how many to compute

They should take the final form of a geodataframe containing the study area features' geometry with additional attributes like courtyard_index_mean, courtyard_index_median, etc, but the values should be saved into compact formats in the cache under the run name categorized by their region_id and loaded if they already exist so it can pick up where it left off if stopped in the middle.

Keep every metric's computation source code as a function in its own file for organizational purposes. Add an explanation of each metric in its docstring.