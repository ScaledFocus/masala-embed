#!/usr/bin/env python3
"""
Synthetic Data Migration Script

This script migrates approved synthetic query data from MLflow experiments to the
database.
It handles the complete flow:
1. Query MLflow for approved runs (data_status = "approved")
2. Download and process CSV data
3. Create/find labeler entries for synthetic generation
4. Insert queries, examples, and labels into database
5. Mark MLflow runs as "migrated" (or "migrated_partial" if --bridged flag used)

Usage:
    python migrate_synthetic_data.py --experiment-name initial-generation-E --dry-run
    python migrate_synthetic_data.py --run-id abc123 --confirm
    python migrate_synthetic_data.py --all-approved --confirm
    python migrate_synthetic_data.py --run-id abc123 --bridged --confirm
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import mlflow
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add project paths
project_root = os.getenv("root_folder")
if project_root:
    sys.path.append(os.path.join(project_root, "esci-dataset", "src"))
    sys.path.append(project_root)

from database.utils.db_utils import get_db_connection  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("migrate_synthetic_data.log"),
    ],
)
logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """Setup MLflow tracking URI."""
    mlflow_tracking_uri = (
        os.path.join(project_root, "mlruns") if project_root else "./mlruns"
    )
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")


def get_approved_runs(experiment_name: str = None, run_id: str = None) -> list[dict]:
    """Get MLflow runs that are approved for migration."""
    if run_id:
        # Get specific run
        run = mlflow.get_run(run_id)
        runs = [run]
    elif experiment_name:
        # Get runs from specific experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_name}")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.data_status = 'approved'",
            output_format="list",
        )
    else:
        # Get all approved runs across experiments
        # Note: MLflow's global tag search doesn't work reliably, so we search
        # each experiment
        logger.info("Searching for approved runs across all experiments...")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        runs = []
        for experiment in experiments:
            # Search for approved runs in this experiment
            exp_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.data_status = 'approved'",
                output_format="list",
                max_results=100,
            )

            if exp_runs:
                logger.info(
                    f"Found {len(exp_runs)} approved runs in experiment: "
                    f"{experiment.name}"
                )
                runs.extend(exp_runs)

    approved_runs = []
    for run in runs:
        # Check if already fully migrated
        data_status = run.data.tags.get("data_status")
        if data_status == "migrated":
            logger.info(f"Skipping already fully migrated run: {run.info.run_id}")
            continue

        # Log partial migration status but allow reprocessing
        if data_status == "migrated_partial":
            logger.info(f"Run {run.info.run_id} has partial migration (bridged only)")

        # Extract git commit hash from MLflow metadata
        # Priority: 1) data_gen_hash param, 2) mlflow.source.git.commit tag
        data_gen_hash = run.data.params.get("data_gen_hash")
        if not data_gen_hash:
            data_gen_hash = run.data.tags.get("mlflow.source.git.commit")

        # Validate git hash exists
        if not data_gen_hash:
            logger.warning(
                f"No git hash found in MLflow metadata for run {run.info.run_id}. "
                "Skipping this run as it cannot be properly tracked."
            )
            continue

        approved_runs.append(
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "data_gen_hash": data_gen_hash,
                "mlflow_run_id": run.data.params.get("mlflow_run_id", run.info.run_id),
                "generation_approach": run.data.params.get("generation_approach"),
                "step": run.data.params.get("step"),
                "output_type": run.data.params.get("output_type"),
                "esci_label": run.data.params.get("esci_label"),
                "tags": run.data.tags,
                "run_name": run.info.run_name,
            }
        )

    return approved_runs


def download_run_data(
    run_id: str, generation_approach: str = None, bridged_only: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Download CSV data from MLflow run artifacts."""
    client = mlflow.tracking.MlflowClient()

    # Determine CSV path based on generation approach
    if generation_approach == "intent":
        # Both 2-step and 3-step intent generation put CSV in outputs/queries_path
        artifacts = client.list_artifacts(run_id, path="outputs/queries_path")
        csv_files = [a for a in artifacts if a.path.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV files found in intent generation run {run_id}")
    else:
        # Initial generation: look directly in outputs
        artifacts = client.list_artifacts(run_id, path="outputs")
        csv_files = [a for a in artifacts if a.path.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV files found in run {run_id}")

    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in run {run_id}")

    # Download the CSV file
    csv_path = csv_files[0].path
    local_path = client.download_artifacts(run_id, csv_path)

    # Load the CSV
    df = pd.read_csv(local_path)
    original_count = len(df)
    logger.info(f"Downloaded {original_count} records from {csv_path}")

    # Initialize filtering metadata
    filter_info = {
        "original_count": original_count,
        "final_count": original_count,
        "has_query_type": "query_type" in df.columns,
        "bridged_requested": bridged_only,
        "filtering_applied": False,
        "can_mark_partial": False,
    }

    # Apply query_type filtering if requested
    if bridged_only:
        if "query_type" in df.columns:
            df = df[df["query_type"] == "bridged"].copy()
            filtered_count = len(df)
            filter_info.update(
                {
                    "final_count": filtered_count,
                    "filtering_applied": True,
                    "can_mark_partial": filtered_count
                    > 0,  # Only if we actually have bridged records
                }
            )

            logger.info(
                f"Filtered to {filtered_count} 'bridged' records "
                f"({original_count - filtered_count} 'intent' records excluded)"
            )

            if filtered_count == 0:
                logger.warning("No 'bridged' query_type records found in the data")
        else:
            # Fail early if --bridged is used but no query_type column exists
            raise ValueError(
                "--bridged flag specified but 'query_type' column not found in data. "
                "Cannot perform partial migration without query_type column. "
                "Remove --bridged flag for full migration or use data with query_type "
                "column."
            )

    return df, filter_info


def ensure_labeler_exists(data_gen_hash: str, generation_approach: str) -> int:
    """Create or find labeler entry for synthetic data generation."""
    labeler_name = data_gen_hash
    labeler_type = f"synthetic_data_generation_{generation_approach}"

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Check if labeler already exists
            cursor.execute(
                "SELECT id FROM labeler WHERE name = %s AND type = %s",
                (labeler_name, labeler_type),
            )
            result = cursor.fetchone()

            if result:
                labeler_id = result[0]
                logger.info(f"Found existing labeler: {labeler_id} ({labeler_name})")
            else:
                # Create new labeler
                cursor.execute(
                    """
                    INSERT INTO labeler (name, role, type)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (labeler_name, "labeler", labeler_type),
                )
                labeler_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Created new labeler: {labeler_id} ({labeler_name})")

    return labeler_id


def validate_consumables_exist(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Validate that referenced consumables exist in database."""
    if "consumable_id" not in df.columns:
        raise ValueError("'consumable_id' column not found in data")

    candidate_ids = df["consumable_id"].unique().tolist()

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Check which consumables exist
            cursor.execute(
                "SELECT id FROM consumable WHERE id = ANY(%s)", (candidate_ids,)
            )
            existing_ids = [
                str(row[0]) for row in cursor.fetchall()
            ]  # Convert to strings for consistency

    missing_ids = [str(cid) for cid in candidate_ids if str(cid) not in existing_ids]

    return existing_ids, missing_ids


def process_enhanced_json_data(
    df: pd.DataFrame,
    data_gen_hash: str,
    mlflow_run_id: str,
    labeler_id: int,
    esci_label: str,
) -> tuple[int, int, int]:
    """Process enhanced JSON format data and insert into database."""

    # Validate consumables exist
    existing_ids, missing_ids = validate_consumables_exist(df)
    if missing_ids:
        logger.warning(
            f"Missing consumables: {missing_ids[:5]}"
            f"{'...' if len(missing_ids) > 5 else ''}"
        )
        # Filter out missing consumables
        df = df[df["consumable_id"].isin(existing_ids)].copy()
        logger.info(f"Filtered to {len(df)} records with existing consumables")

    # Group by unique queries to avoid duplicates
    query_groups = df.groupby("query")

    queries_inserted = 0
    examples_inserted = 0
    labels_inserted = 0

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for query_text, group_df in tqdm(query_groups, desc="Processing queries"):
                # Parse query filters from first record in group
                first_record = group_df.iloc[0]

                # Build query_filters JSON from enhanced format
                query_filters = {}

                # Parse query_filters if it exists
                if "query_filters" in first_record and pd.notna(
                    first_record["query_filters"]
                ):
                    try:
                        query_filters = json.loads(first_record["query_filters"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Check if query already exists, if not insert new one
                cursor.execute(
                    """
                    SELECT id FROM query
                    WHERE query_content = %s
                    LIMIT 1
                    """,
                    (query_text,),
                )
                existing_query = cursor.fetchone()

                if existing_query:
                    # Use existing query
                    query_id = existing_query[0]
                else:
                    # Insert new query
                    cursor.execute(
                        """
                        INSERT INTO query (query_content, query_filters, data_gen_hash,
                                         mlflow_run_id)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            query_text,
                            json.dumps(query_filters) if query_filters else None,
                            data_gen_hash,
                            mlflow_run_id,
                        ),
                    )
                    query_id = cursor.fetchone()[0]
                    queries_inserted += 1

                # Insert examples and labels for each candidate in this query group
                for _, row in group_df.iterrows():
                    candidate_id = str(row["consumable_id"])

                    # Skip if consumable doesn't exist
                    if candidate_id not in existing_ids:
                        continue

                    # Insert example (query-consumable pair)
                    cursor.execute(
                        """
                        INSERT INTO example (query_id, consumable_id, example_gen_hash)
                        VALUES (%s, %s, %s)
                        RETURNING id
                        """,
                        (query_id, candidate_id, data_gen_hash),
                    )
                    example_id = cursor.fetchone()[0]
                    examples_inserted += 1

                    # Insert label
                    cursor.execute(
                        """
                        INSERT INTO label (labeler_id, example_id, esci_label,
                                         auto_label_score)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            labeler_id,
                            example_id,
                            esci_label,
                            1.0,
                        ),  # 1.0 score for synthetic data
                    )
                    labels_inserted += 1

            conn.commit()

    return queries_inserted, examples_inserted, labels_inserted


def mark_run_as_migrated(run_id: str, partial: bool = False) -> None:
    """Mark MLflow run as successfully migrated."""
    client = mlflow.tracking.MlflowClient()

    if partial:
        status = "migrated_partial"
        logger.info(
            f"Marked run {run_id} as migrated_partial (only bridged records migrated)"
        )
    else:
        status = "migrated"
        logger.info(f"Marked run {run_id} as migrated")

    client.set_tag(run_id, "data_status", status)
    client.set_tag(run_id, "migration_timestamp", datetime.now().isoformat())

    if partial:
        client.set_tag(run_id, "migration_type", "bridged_only")


def migrate_run(
    run_data: dict, dry_run: bool = True, bridged_only: bool = False
) -> dict:
    """Migrate a single MLflow run to database."""
    run_id = run_data["run_id"]
    data_gen_hash = run_data["data_gen_hash"]
    mlflow_run_id = run_data["mlflow_run_id"]
    generation_approach = run_data["generation_approach"]
    output_type = run_data["output_type"]
    esci_label = run_data["esci_label"]

    # Validate git hash from MLflow metadata
    if not data_gen_hash:
        raise ValueError(
            f"No git hash available for run {run_id}. "
            "Cannot maintain data traceability without git hash."
        )

    # For intent generation, default to "E" (Exact match) if esci_label is None
    if generation_approach == "intent" and esci_label is None:
        esci_label = "E"
        logger.info("Defaulting to ESCI label 'E' for intent generation")

    logger.info(f"Processing run: {run_id} ({run_data['run_name']})")
    logger.info(
        f"  Approach: {generation_approach}, Output: {output_type}, ESCI: {esci_label}"
    )
    logger.info(f"  Using git hash from MLflow: {data_gen_hash}")

    try:
        # Download data
        df, filter_info = download_run_data(run_id, generation_approach, bridged_only)

        # Check if we have any data to process
        if len(df) == 0:
            logger.error(f"No records to process for run {run_id}")
            return {
                "run_id": run_id,
                "status": "error",
                "error": "No records found after filtering",
                "records_processed": 0,
                "queries_inserted": 0,
                "examples_inserted": 0,
                "labels_inserted": 0,
            }

        if dry_run:
            logger.info(f"[DRY RUN] Would process {len(df)} records")
            if filter_info["bridged_requested"] and filter_info["can_mark_partial"]:
                logger.info("[DRY RUN] Would mark as 'migrated_partial' (bridged only)")
            elif (
                filter_info["bridged_requested"] and not filter_info["can_mark_partial"]
            ):
                logger.info(
                    "[DRY RUN] Would NOT mark as migrated "
                    "(no valid records after filtering)"
                )
            else:
                logger.info("[DRY RUN] Would mark as 'migrated' (full migration)")

            return {
                "run_id": run_id,
                "status": "dry_run",
                "records_processed": len(df),
                "queries_inserted": 0,
                "examples_inserted": 0,
                "labels_inserted": 0,
            }

        # Ensure labeler exists
        labeler_id = ensure_labeler_exists(data_gen_hash, generation_approach)

        # Process data based on output type
        if output_type in ["enhanced_json", "intent_matches"]:
            queries_inserted, examples_inserted, labels_inserted = (
                process_enhanced_json_data(
                    df, data_gen_hash, mlflow_run_id, labeler_id, esci_label
                )
            )
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        # Mark as migrated based on what we actually processed
        if filter_info["bridged_requested"] and filter_info["can_mark_partial"]:
            # We used --bridged flag and actually migrated some bridged records
            mark_run_as_migrated(run_id, partial=True)
        elif not filter_info["bridged_requested"]:
            # Full migration (no --bridged flag)
            mark_run_as_migrated(run_id, partial=False)
        else:
            # --bridged was requested but we can't mark as partial
            # (either no query_type column or no bridged records found)
            logger.warning(
                f"Not marking run {run_id} as migrated - insufficient data for "
                f"partial migration"
            )
            return {
                "run_id": run_id,
                "status": "error",
                "error": "Cannot mark as migrated - no valid bridged records found",
                "records_processed": len(df),
                "queries_inserted": queries_inserted,
                "examples_inserted": examples_inserted,
                "labels_inserted": labels_inserted,
            }

        return {
            "run_id": run_id,
            "status": "success",
            "records_processed": len(df),
            "queries_inserted": queries_inserted,
            "examples_inserted": examples_inserted,
            "labels_inserted": labels_inserted,
        }

    except Exception as e:
        logger.error(f"Failed to migrate run {run_id}: {e}")
        return {
            "run_id": run_id,
            "status": "error",
            "error": str(e),
            "records_processed": 0,
            "queries_inserted": 0,
            "examples_inserted": 0,
            "labels_inserted": 0,
        }


def main():
    """Main migration execution."""
    parser = argparse.ArgumentParser(
        description="Migrate approved synthetic data to database"
    )
    parser.add_argument("--experiment-name", help="MLflow experiment name to migrate")
    parser.add_argument("--run-id", help="Specific MLflow run ID to migrate")
    parser.add_argument(
        "--all-approved", action="store_true", help="Migrate all approved runs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without making changes"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm migration (required for actual migration)",
    )
    parser.add_argument(
        "--bridged",
        action="store_true",
        help="Only migrate records with query_type='bridged' "
        "(for intent generation data)",
    )

    args = parser.parse_args()

    if not any([args.experiment_name, args.run_id, args.all_approved]):
        parser.error("Must specify --experiment-name, --run-id, or --all-approved")

    if not args.dry_run and not args.confirm:
        parser.error(
            "Must use --confirm flag for actual migration (or --dry-run to preview)"
        )

    try:
        # Setup MLflow
        setup_mlflow()

        # Get approved runs
        logger.info("=" * 60)
        logger.info("SYNTHETIC DATA MIGRATION STARTED")
        logger.info("=" * 60)

        if args.run_id:
            approved_runs = get_approved_runs(run_id=args.run_id)
        elif args.experiment_name:
            approved_runs = get_approved_runs(experiment_name=args.experiment_name)
        else:
            approved_runs = get_approved_runs()

        if not approved_runs:
            logger.info("No approved runs found for migration")
            return

        logger.info(f"Found {len(approved_runs)} approved runs for migration")

        if args.bridged:
            logger.info(
                "🔗 BRIDGED MODE: Only migrating records with query_type='bridged'"
            )
            logger.info(
                "   Runs will be marked as 'migrated_partial' instead of 'migrated'"
            )

        # Migrate each run
        results = []
        for run_data in approved_runs:
            result = migrate_run(
                run_data, dry_run=args.dry_run, bridged_only=args.bridged
            )
            results.append(result)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)

        successful = [r for r in results if r["status"] == "success"]
        errors = [r for r in results if r["status"] == "error"]
        dry_runs = [r for r in results if r["status"] == "dry_run"]

        if dry_runs:
            total_records = sum(r["records_processed"] for r in dry_runs)
            logger.info(
                f"[DRY RUN] Would migrate {len(dry_runs)} runs with "
                f"{total_records} total records"
            )
        else:
            logger.info(f"Successfully migrated: {len(successful)} runs")
            logger.info(f"Failed migrations: {len(errors)} runs")

            if successful:
                total_queries = sum(r["queries_inserted"] for r in successful)
                total_examples = sum(r["examples_inserted"] for r in successful)
                total_labels = sum(r["labels_inserted"] for r in successful)

                logger.info(
                    f"Total inserted: {total_queries} queries, "
                    f"{total_examples} examples, {total_labels} labels"
                )

        # Show errors
        for error_result in errors:
            logger.error(f"Run {error_result['run_id']}: {error_result['error']}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
