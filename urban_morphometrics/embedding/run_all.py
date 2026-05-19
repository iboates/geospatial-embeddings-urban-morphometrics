"""
Run all experiments from config files in the configs folder.

Automatically discovers all YAML config files (except base.yaml) in the configs
directory and runs each experiment 10 times with predefined random seeds.

Usage:
    python run_all.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from run_experiment import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Predefined random seeds for reproducible experiments
RANDOM_SEEDS = [42, 54, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

DATASETS = ["HouseSalesInKingCounty", "ChicagoCrime", "PhiladelphiaCrime"]

RESOLUTIONS = [8, 9]


def run_all() -> None:
    """Discover and run all experiments from config files with multiple seeds."""
    configs_dir = Path(__file__).parent / "configs"

    if not configs_dir.exists():
        logger.error("Configs directory not found: %s", configs_dir)
        return

    # Find all YAML config files except base.yaml
    config_files = sorted(
        [f for f in configs_dir.glob("*.yaml") if f.name != "base.yaml"]
    )

    if not config_files:
        logger.warning("No config files found in %s (excluding base.yaml)", configs_dir)
        return

    total_runs = (
        len(config_files) * len(RANDOM_SEEDS) * len(DATASETS) * len(RESOLUTIONS)
    )
    logger.info(
        "Found %d config file(s) - will run each with %d seeds = %d total runs",
        len(config_files),
        len(RANDOM_SEEDS),
        total_runs,
    )

    results = {}
    run_count = 0

    for resolution in RESOLUTIONS:
        for dataset_name in DATASETS:
            for config_path in config_files:
                results[config_path.name] = {}
                for seed in RANDOM_SEEDS:
                    run_count += 1
                    logger.info("─" * 80)
                    logger.info(
                        "[%d/%d] Running: %s (seed=%d)",
                        run_count,
                        total_runs,
                        config_path.name,
                        seed,
                    )
                    logger.info("─" * 80)

                    try:
                        # Set random seed as environment variable
                        os.environ["RANDOM_SEED"] = str(seed)
                        pl.seed_everything(seed, workers=True)

                        result = run(
                            str(config_path),
                            resolution=resolution,
                            dataset_name=dataset_name,
                        )
                        results[config_path.name][seed] = {
                            "status": "success",
                            "result": result,
                        }
                        logger.info(
                            "✓ Completed: %s (seed=%d)",
                            config_path.name,
                            seed,
                        )
                    except Exception as e:
                        logger.error(
                            "✗ Failed: %s (seed=%d) - %s",
                            config_path.name,
                            seed,
                            e,
                            exc_info=True,
                        )
                        results[config_path.name][seed] = {
                            "status": "failed",
                            "error": str(e),
                        }

    # Summary
    logger.info("─" * 80)
    logger.info("=== SUMMARY ===")
    logger.info("─" * 80)

    total_successful = 0
    total_failed = 0

    for config_name, seed_results in results.items():
        successful = sum(1 for r in seed_results.values() if r["status"] == "success")
        failed = sum(1 for r in seed_results.values() if r["status"] == "failed")
        total_successful += successful
        total_failed += failed

        logger.info(
            "%s - Success: %d/%d | Failed: %d/%d",
            config_name,
            successful,
            len(RANDOM_SEEDS),
            failed,
            len(RANDOM_SEEDS),
        )

    logger.info("─" * 80)
    logger.info(
        "Total: %d | Success: %d | Failed: %d",
        total_runs,
        total_successful,
        total_failed,
    )
    logger.info("─" * 80)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_all()
