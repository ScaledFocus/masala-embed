#!/usr/bin/env python3
"""
MLflow Runs Summary Script

This script generates a CSV summary of all migrated MLflow runs (both full and partial)
across experiments, extracting key parameters like start_idx, limit, generation type,
and migration status.

Usage:
    python scripts/mlflow_runs_summary.py --output summary.csv
    python scripts/mlflow_runs_summary.py --experiment-name "Intent_Generation" \\
        --output intent_summary.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add project paths
project_root = os.getenv("root_folder")
if project_root:
    sys.path.append(os.path.join(project_root, "esci-dataset", "src"))
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """Setup MLflow tracking URI."""
    mlflow_tracking_uri = (
        os.path.join(project_root, "mlruns") if project_root else "./mlruns"
    )
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


def determine_generation_type(params: dict, tags: dict) -> str:
    """Determine the generation type based on parameters and tags."""
    # Check script name first
    script_name = params.get("script_name", "")

    if script_name == "initial_generation.py":
        return "initial_generation"
    elif script_name == "intent_generation_approach.py":
        # Check if it stopped at intents (2-step) or completed (3-step)
        stop_at_intents = params.get("stop_at_intents", False)
        if stop_at_intents or params.get("step", "") == "1_2":
            return "intent_generation_2step"
        else:
            return "intent_generation_3step"

    # Fallback: try to infer from tags and other parameters
    if (
        "initial-generation" in tags.keys()
        or "initial_generation" in str(tags.values()).lower()
    ):
        return "initial_generation"
    elif (
        "intent-generation" in tags.keys()
        or "intent_generation" in str(tags.values()).lower()
    ):
        stop_at_intents = params.get("stop_at_intents", False)
        if stop_at_intents:
            return "intent_generation_2step"
        else:
            return "intent_generation_3step"

    return "unknown"


def get_migrated_runs_summary(experiment_name: str = None) -> pd.DataFrame:
    """Get summary of all migrated runs (both full and partial)."""
    logger.info("Searching for migrated MLflow runs (full and partial)...")

    summary_data = []

    if experiment_name:
        # Get specific experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_name}")
            return pd.DataFrame()
        experiments = [experiment]
        logger.info(f"Analyzing experiment: {experiment_name}")
    else:
        # Get all experiments
        experiments = mlflow.search_experiments()
        logger.info(f"Analyzing {len(experiments)} experiments")

    for experiment in experiments:
        logger.info(
            f"Processing experiment: {experiment.name} (ID: {experiment.experiment_id})"
        )

        try:
            # Search for migrated runs (both full and partial) in this experiment
            # MLflow doesn't support OR in filter strings, so we need to search
            # separately
            migrated_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.data_status = 'migrated'",
                output_format="pandas",
            )

            migrated_partial_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.data_status = 'migrated_partial'",
                output_format="pandas",
            )

            # Combine the results
            runs = pd.concat([migrated_runs, migrated_partial_runs], ignore_index=True)

            if runs.empty:
                logger.info(f"No migrated runs found in experiment: {experiment.name}")
                continue

            logger.info(
                f"Found {len(runs)} migrated runs (full and partial) in experiment: "
                f"{experiment.name}"
            )

            for _, run in runs.iterrows():
                # Extract parameters from the params columns
                params = {}
                tags = {}

                # Get all param columns (they start with 'params.')
                param_cols = [col for col in runs.columns if col.startswith("params.")]
                for col in param_cols:
                    param_name = col.replace("params.", "")
                    params[param_name] = run[col]

                # Get all tag columns (they start with 'tags.')
                tag_cols = [col for col in runs.columns if col.startswith("tags.")]
                for col in tag_cols:
                    tag_name = col.replace("tags.", "")
                    tags[tag_name] = run[col]

                # Extract key information
                run_id = run["run_id"]
                start_time = run["start_time"]
                end_time = run["end_time"]

                # Extract parameters with defaults
                start_idx = int(params.get("start_idx", 0))
                limit = params.get("limit", "None")
                if limit and limit != "None":
                    limit = int(limit)
                else:
                    limit = None

                batch_size = params.get("batch_size", "None")
                if batch_size and batch_size != "None":
                    batch_size = int(batch_size)
                else:
                    batch_size = None

                model = params.get("model", "unknown")
                generation_type = determine_generation_type(params, tags)

                # Determine migration status
                migration_status = tags.get("data_status", "unknown")

                # Extract ESCI label
                esci_label = params.get("esci_label", "unknown")

                # For Intent generation experiments, default to "E" if no
                # esci_label is found
                if esci_label == "unknown" and experiment.name.startswith("Intent"):
                    esci_label = "E"

                # Extract git hash (data generation hash)
                # Priority: 1) data_gen_hash param, 2) mlflow.source.git.commit tag
                data_gen_hash = params.get("data_gen_hash")
                if not data_gen_hash:
                    data_gen_hash = tags.get("mlflow.source.git.commit", "unknown")

                # Extract metrics
                total_queries = run.get("metrics.total_queries_generated", 0)
                unique_queries = run.get("metrics.unique_queries_generated", 0)
                successful_matches = run.get("metrics.successful_matches", 0)
                runtime_seconds = run.get("metrics.total_runtime_seconds", 0)

                # Calculate end_idx based on start_idx and limit
                end_idx = None
                if limit is not None:
                    end_idx = start_idx + limit - 1

                summary_record = {
                    "experiment_name": experiment.name,
                    "run_id": run_id,
                    "generation_type": generation_type,
                    "migration_status": migration_status,
                    "esci_label": esci_label,
                    "data_gen_hash": data_gen_hash,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "limit": limit,
                    "batch_size": batch_size,
                    "model": model,
                    "total_queries_generated": total_queries,
                    "unique_queries_generated": unique_queries,
                    "successful_matches": successful_matches,
                    "runtime_seconds": runtime_seconds,
                    "start_time": start_time,
                    "end_time": end_time,
                }

                summary_data.append(summary_record)

        except Exception as e:
            logger.error(f"Error processing experiment {experiment.name}: {e}")
            continue

    if not summary_data:
        logger.warning("No migrated runs found across all experiments")
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_data)

    # Sort by experiment_name, then by start_idx
    summary_df = summary_df.sort_values(["experiment_name", "start_idx"])

    logger.info(f"Generated summary for {len(summary_df)} migrated runs")
    return summary_df


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate CSV summary of migrated MLflow runs "
        "(both full and partial)"
    )
    parser.add_argument(
        "--experiment-name",
        help="Specific experiment name to analyze (default: all experiments)",
    )
    parser.add_argument(
        "--output",
        default="mlflow_migrated_runs_summary.csv",
        help="Output CSV file path (default: mlflow_migrated_runs_summary.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/summary",
        help="Output directory (default: output/summary)",
    )

    args = parser.parse_args()

    # Setup MLflow
    setup_mlflow()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate summary
    summary_df = get_migrated_runs_summary(args.experiment_name)

    if summary_df.empty:
        logger.warning("No migrated runs found. No summary file generated.")
        return 1

    # Add timestamp to filename if no specific path provided
    if args.output == "mlflow_migrated_runs_summary.csv":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"mlflow_migrated_runs_summary_{timestamp}.csv"
    else:
        output_file = output_dir / args.output

    # Save summary
    summary_df.to_csv(output_file, index=False)
    logger.info(f"Summary saved to: {output_file}")

    # Also save a copy to database/ directory with fixed filename
    database_output = Path("database") / "synthetic_data_migration_summary.csv"
    database_output.parent.mkdir(exist_ok=True)
    summary_df.to_csv(database_output, index=False)
    logger.info(f"Copy saved to: {database_output}")

    # Print summary statistics
    print("\n📊 MLflow Migrated Runs Summary")
    print("=" * 50)
    print(f"Total migrated runs: {len(summary_df)}")
    print(f"Experiments: {summary_df['experiment_name'].nunique()}")
    print(f"Generation types: {summary_df['generation_type'].value_counts().to_dict()}")
    print(
        f"Migration statuses: {summary_df['migration_status'].value_counts().to_dict()}"
    )
    print(f"Total queries generated: {summary_df['total_queries_generated'].sum():,}")
    print(f"Total unique queries: {summary_df['unique_queries_generated'].sum():,}")
    date_range = f"{summary_df['start_time'].min()} to {summary_df['start_time'].max()}"
    print(f"Date range: {date_range}")
    print(f"\n💾 Summary saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
