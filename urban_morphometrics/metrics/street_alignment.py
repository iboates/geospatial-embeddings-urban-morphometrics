"""Street alignment metric.

Deviation between each building's orientation and the orientation of its
nearest street segment. Low values indicate buildings parallel or perpendicular
to the street grid; high values indicate organic or irregular placement.

Each building is matched to its nearest street using momepy.get_nearest_street.
Buildings farther than street_alignment_max_distance (configurable via
MetricConfig, default 500 m) from any street receive a NaN value and are
excluded from cell-level aggregation.

Requires neighbourhood context so both buildings and streets near the cell
boundary can find their true nearest counterparts.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("street_alignment")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Deviation between each building's orientation and its nearest street (°)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "street_alignment", num_quantiles)

    streets = ctx.focal_plus_neighbourhood_vehicle_highways
    if streets.empty:
        return aggregate_series(pd.Series(dtype=float), "street_alignment", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    max_dist = ctx.config.street_alignment_max_distance

    nearest = momepy.get_nearest_street(all_b, streets, max_distance=max_dist)
    building_orient = momepy.orientation(all_b)
    street_orient = momepy.orientation(streets)

    valid = nearest.notna()
    values = momepy.street_alignment(building_orient[valid], street_orient, nearest[valid])
    focal_values = values.reindex(b.index)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(street_alignment=focal_values), features_dir / "street_alignment.gpkg")
    return aggregate_series(focal_values, "street_alignment", num_quantiles)
