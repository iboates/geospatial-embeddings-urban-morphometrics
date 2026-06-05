"""Urban morphometrics pipeline — importable function and CLI entry point."""

import argparse
import logging
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metric_config import MetricConfig
from urban_morphometrics.metrics import compute_metrics
from urban_morphometrics.osm_loader import load_osm_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _resolve_pbf(pbf: str | Path, output_folder: Path) -> Path:
    """Return a local path to the PBF file, downloading it first if a URL is given.

    Downloaded files are cached in {output_folder}/pbf_cache/ keyed by filename.
    Re-specifying the same URL will reuse the cached file without re-downloading.
    """
    pbf = str(pbf)
    parsed = urllib.parse.urlparse(pbf)
    if parsed.scheme in ("http", "https"):
        filename = Path(parsed.path).name
        if not filename.endswith(".pbf"):
            raise ValueError(f"URL does not point to a .pbf file: {pbf}")
        pbf_cache_dir = output_folder / "pbf_cache"
        pbf_cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = pbf_cache_dir / filename
        if local_path.exists():
            log.info("Using cached PBF: %s", local_path)
        else:
            log.info("Downloading PBF from %s", pbf)
            _download_with_progress(pbf, local_path)
        return local_path

    local_path = Path(pbf)
    if not local_path.exists():
        raise FileNotFoundError(f"PBF file not found: {local_path}")
    return local_path


def _download_with_progress(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            chunk = 1024 * 1024  # 1 MB
            with (
                tmp.open("wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                    leave=True,
                ) as bar,
            ):
                while True:
                    buf = response.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    bar.update(len(buf))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def compute_urban_morphometrics(
    study_area_gdf: gpd.GeoDataFrame,
    pbf_path: str | Path | None,
    run_name: str,
    output_folder: str | Path,
    neighbourhood_distance: float = 500.0,
    num_quantiles: int = 10,
    metrics: list[str] | None = None,
    equal_area_crs: str = "EPSG:3395",
    equidistant_crs: str = "EPSG:4087",
    conformal_crs: str = "EPSG:3857",
    debug: bool = False,
    use_cache: bool = True,
    metric_config: "dict | MetricConfig | None" = None,
    export_features: bool = False,
    n_workers: int = 1,
) -> gpd.GeoDataFrame:
    """Compute urban morphology metrics for a study area from OSM data.

    Args:
        study_area_gdf: GeoDataFrame of polygons in WGS84 with 'region_id' as its index.
        pbf_path: Path or URL to a .pbf OSM dump file. URLs are downloaded once and
            cached in {output_folder}/pbf_cache/ for reuse across runs.
        run_name: Name of this pipeline run (used for output and cache folders).
        output_folder: Root directory for all outputs.
        neighbourhood_distance: Buffer distance (metres) around each cell used
            to gather neighbourhood context for edge-sensitive metrics.
        num_quantiles: Number of quantile bands to compute for per-feature metrics.
        metrics: List of metric names to compute. None or empty list means all metrics.
        equal_area_crs: CRS string for equal-area projections.
        equidistant_crs: CRS string for equidistant projections.
        conformal_crs: CRS string for conformal projections.
        debug: When True, dump intermediate layers to the debug folder.
        use_cache: When False, metric results are always recomputed even if a cached
            ``_metrics.parquet`` exists. The cache is also not written.
        export_features: When True, write per-feature GeoPackages to features/{region_id}/
            for each metric.
        n_workers: Number of threads for parallel cell processing (default 1 = sequential).
            Shapely/numpy release the GIL so threading provides real parallelism without
            the overhead of copying osm_data across processes.
    """
    output_folder = Path(output_folder)

    if study_area_gdf.index.name != "region_id":
        raise ValueError("study_area_gdf must have 'region_id' as its index")

    if pbf_path is not None:
        pbf_path = _resolve_pbf(pbf_path, output_folder)
    metrics = metrics or ["all"]

    from urban_morphometrics.metric_config import MetricConfig

    if metric_config is None:
        cfg = MetricConfig()
    elif isinstance(metric_config, dict):
        cfg = MetricConfig.from_dict(metric_config)
    elif isinstance(metric_config, MetricConfig):
        cfg = metric_config
    else:
        raise TypeError(
            f"metric_config must be a dict, MetricConfig, or None; got {type(metric_config).__name__}"
        )

    run_dir = output_folder / run_name
    cache_dir = run_dir / "cache"
    results_dir = run_dir / "results"
    debug_dir = run_dir / "debug"
    features_dir = run_dir / "features"

    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    log.info("Run: %s", run_name)
    log.info("Study area: %d cells", len(study_area_gdf))
    log.info("PBF: %s", pbf_path)
    log.info("Output: %s", run_dir)
    log.info("Neighbourhood distance: %sm", neighbourhood_distance)
    log.info("Quantiles: %s", num_quantiles)
    log.info("Equal-area CRS: %s", equal_area_crs)
    log.info("Equidistant CRS: %s", equidistant_crs)
    log.info("Conformal CRS: %s", conformal_crs)
    log.info("Debug: %s", debug)
    log.info("Metrics (%d): %s", len(metrics), ", ".join(metrics))
    log.info("Metric config: %s", cfg.to_dict())

    osm_data = load_osm_data(pbf_path, study_area_gdf)

    if debug:
        log.info("Writing debug OSM layers...")
        osm_data.buildings.to_file(debug_dir / "buildings.gpkg", driver="GPKG")
        osm_data.highways.to_file(debug_dir / "highways.gpkg", driver="GPKG")
        osm_data.landuse.to_file(debug_dir / "landuse.gpkg", driver="GPKG")
        osm_data.water.to_file(debug_dir / "water.gpkg", driver="GPKG")
        osm_data.pedestrian_areas.to_file(
            debug_dir / "pedestrian_areas.gpkg", driver="GPKG"
        )

    def _process_cell(region_id, row):
        cell_cache_dir = cache_dir / str(region_id)
        metrics_cache = cell_cache_dir / "_metrics.parquet"
        if use_cache and metrics_cache.exists():
            return (
                region_id,
                row.geometry,
                pd.read_parquet(metrics_cache).iloc[0].to_dict(),
            )
        cell_features_dir = (features_dir / str(region_id)) if export_features else None
        ctx = CellContext(
            region_id=region_id,
            cell_geometry=row.geometry,
            osm_data=osm_data,
            neighbourhood_distance=neighbourhood_distance,
            equal_area_crs=equal_area_crs,
            equidistant_crs=equidistant_crs,
            conformal_crs=conformal_crs,
            cache_dir=cell_cache_dir,
            config=cfg,
            features_dir=cell_features_dir,
        )
        metric_row = compute_metrics(
            ctx, metrics, num_quantiles, features_dir=cell_features_dir
        )
        if use_cache:
            pd.DataFrame([metric_row]).to_parquet(metrics_cache)
        return region_id, row.geometry, metric_row

    rows = []
    n_total = len(study_area_gdf)
    n_failed = 0

    out_path = results_dir / "metrics.gpkg"
    if use_cache and out_path.exists():
        results_gdf = gpd.read_file(out_path)
        results_gdf = results_gdf.set_index("region_id")
        log.info("Results loaded from %s", out_path)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_cell, region_id, row): region_id
                for region_id, row in study_area_gdf.iterrows()
            }
            for future in tqdm(
                as_completed(futures), total=n_total, desc="Cells", unit="cell"
            ):
                region_id = futures[future]
                try:
                    rid, geom, metric_row = future.result()
                    rows.append({"region_id": rid, "geometry": geom, **metric_row})
                except Exception:
                    log.warning("Cell %s failed — skipping", region_id, exc_info=True)
                    n_failed += 1

        if n_failed:
            log.warning(
                "%d / %d cells failed and were excluded from results", n_failed, n_total
            )

        results_gdf = gpd.GeoDataFrame(rows, crs=study_area_gdf.crs).set_index(
            "region_id"
        )
        results_gdf.to_file(out_path, driver="GPKG")
        log.info("Results written to %s", out_path)
    return results_gdf


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="urban-morphometrics",
        description="Compute urban morphology metrics for a study area from OSM data.",
    )
    p.add_argument(
        "study_area_path",
        help="Path to study area file (GeoPackage/GeoJSON/Shapefile) in WGS84 with 'region_id' column or index.",
    )
    p.add_argument(
        "pbf_path",
        help="Path or URL to a .pbf OSM dump file. URLs are downloaded and cached in {output_folder}/pbf_cache/.",
    )
    p.add_argument("run_name", help="Name of this pipeline run.")
    p.add_argument("output_folder", help="Root directory for all outputs.")
    p.add_argument(
        "--neighbourhood-distance",
        type=float,
        default=500.0,
        metavar="METRES",
        help="Buffer distance around each cell for neighbourhood context (default: 500).",
    )
    p.add_argument(
        "--num-quantiles",
        type=int,
        default=10,
        metavar="N",
        help="Number of quantile bands for per-feature metrics (default: 10).",
    )
    p.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated metric names, or 'all' (default: all).",
    )
    p.add_argument(
        "--equal-area-crs",
        default="EPSG:3395",
        help="Equal-area CRS (default: EPSG:3395).",
    )
    p.add_argument(
        "--equidistant-crs",
        default="EPSG:4087",
        help="Equidistant CRS (default: EPSG:4087).",
    )
    p.add_argument(
        "--conformal-crs",
        default="EPSG:3857",
        help="Conformal CRS (default: EPSG:3857).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Dump intermediate layers to the debug folder.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute metrics for every cell, ignoring and not writing the cache.",
    )
    p.add_argument(
        "--export-features",
        action="store_true",
        help="Write per-feature GeoPackages to features/{region_id}/ for each metric.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker threads for cell processing (default: 1).",
    )
    p.add_argument(
        "--metric-config",
        metavar="PATH",
        help="Path to a JSON file with metric configuration parameters (knn_k, tessellation_clip, etc.).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    study_area_path = Path(args.study_area_path)
    if not study_area_path.exists():
        log.error("Study area file not found: %s", study_area_path)
        sys.exit(1)

    study_area_gdf = gpd.read_file(study_area_path)
    if study_area_gdf.index.name != "region_id":
        if "region_id" not in study_area_gdf.columns:
            log.error("Study area file must have a 'region_id' column or index.")
            sys.exit(1)
        study_area_gdf = study_area_gdf.set_index("region_id")

    metrics = (
        None
        if args.metrics.strip().lower() == "all"
        else [m.strip() for m in args.metrics.split(",") if m.strip()]
    )

    metric_config = None
    if args.metric_config:
        import json

        with open(args.metric_config) as f:
            metric_config = json.load(f)

    try:
        compute_urban_morphometrics(
            study_area_gdf=study_area_gdf,
            pbf_path=args.pbf_path,
            run_name=args.run_name,
            output_folder=args.output_folder,
            neighbourhood_distance=args.neighbourhood_distance,
            num_quantiles=args.num_quantiles,
            metrics=metrics,
            equal_area_crs=args.equal_area_crs,
            equidistant_crs=args.equidistant_crs,
            conformal_crs=args.conformal_crs,
            debug=args.debug,
            use_cache=not args.no_cache,
            metric_config=metric_config,
            export_features=args.export_features,
            n_workers=args.workers,
        )
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
