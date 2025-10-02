#!/usr/bin/env python3
"""
Synthetic Data Rollback Script

This script rolls back synthetic query data migrations by:
1. Finding migrated MLflow run by run_id
2. Removing associated labels, examples, and queries from database
3. Removing the labeler if no other data references it
4. Resetting MLflow run tag from "migrated" to "pending_approval"

Usage:
    python rollback_synthetic_data.py --run-id abc123 --dry-run
    python rollback_synthetic_data.py --run-id abc123 --confirm
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import mlflow
from dotenv import load_dotenv

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
        logging.FileHandler("rollback_synthetic_data.log"),
    ],
)
logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """Setup MLflow tracking URI."""
    mlflow_tracking_uri = (
        os.path.join(project_root, "mlruns") if project_root else "./mlruns"
    )
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")


def get_migrated_run(run_id: str) -> dict:
    """Get MLflow run that is marked as migrated."""
    run = mlflow.get_run(run_id)

    # Check if run is migrated
    if run.data.tags.get("data_status") != "migrated":
        status = run.data.tags.get("data_status", "unknown")
        raise ValueError(f"Run {run_id} is not marked as migrated. Status: {status}")

    # Extract git commit hash from MLflow metadata
    data_gen_hash = run.data.params.get("data_gen_hash")
    if not data_gen_hash:
        data_gen_hash = run.data.tags.get("mlflow.source.git.commit")

    if not data_gen_hash:
        raise ValueError(f"No git hash found in MLflow metadata for run {run_id}")

    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "data_gen_hash": data_gen_hash,
        "mlflow_run_id": run.data.params.get("mlflow_run_id", run.info.run_id),
        "generation_approach": run.data.params.get("generation_approach"),
        "tags": run.data.tags,
        "run_name": run.info.run_name,
    }


def remove_run_data(
    data_gen_hash: str,
    mlflow_run_id: str,
    generation_approach: str,
    dry_run: bool = True,
) -> dict:
    """Remove all database records associated with a migrated run."""

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Find labeler for this run
            labeler_name = data_gen_hash
            labeler_type = f"synthetic_data_generation_{generation_approach}"

            cursor.execute(
                "SELECT id FROM labeler WHERE name = %s AND type = %s",
                (labeler_name, labeler_type),
            )
            labeler_result = cursor.fetchone()

            if not labeler_result:
                logger.warning(f"No labeler found for {labeler_name} ({labeler_type})")
                return {
                    "queries_removed": 0,
                    "examples_removed": 0,
                    "labels_removed": 0,
                    "labelers_removed": 0,
                }

            labeler_id = labeler_result[0]

            # Find queries created by this run
            cursor.execute(
                "SELECT id FROM query WHERE data_gen_hash = %s AND mlflow_run_id = %s",
                (data_gen_hash, mlflow_run_id),
            )
            query_ids = [row[0] for row in cursor.fetchall()]

            if not query_ids:
                logger.warning(f"No queries found for run {mlflow_run_id}")
                return {
                    "queries_removed": 0,
                    "examples_removed": 0,
                    "labels_removed": 0,
                    "labelers_removed": 0,
                }

            logger.info(f"Found {len(query_ids)} queries to remove")

            if dry_run:
                # Count what would be removed
                cursor.execute(
                    "SELECT COUNT(*) FROM example WHERE query_id = ANY(%s)",
                    (query_ids,),
                )
                example_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COUNT(*) FROM label
                    WHERE labeler_id = %s AND example_id IN (
                        SELECT id FROM example WHERE query_id = ANY(%s)
                    )
                    """,
                    (labeler_id, query_ids),
                )
                label_count = cursor.fetchone()[0]

                logger.info(
                    f"[DRY RUN] Would remove {len(query_ids)} queries, "
                    f"{example_count} examples, {label_count} labels"
                )

                return {
                    "queries_removed": len(query_ids),
                    "examples_removed": example_count,
                    "labels_removed": label_count,
                    "labelers_removed": 1,
                }

            # Remove labels first (foreign key constraints)
            cursor.execute(
                """
                DELETE FROM label
                WHERE labeler_id = %s AND example_id IN (
                    SELECT id FROM example WHERE query_id = ANY(%s)
                )
                """,
                (labeler_id, query_ids),
            )
            labels_removed = cursor.rowcount

            # Remove examples
            cursor.execute("DELETE FROM example WHERE query_id = ANY(%s)", (query_ids,))
            examples_removed = cursor.rowcount

            # Remove queries
            cursor.execute("DELETE FROM query WHERE id = ANY(%s)", (query_ids,))
            queries_removed = cursor.rowcount

            # Check if labeler has any other data, if not remove it
            cursor.execute(
                "SELECT COUNT(*) FROM label WHERE labeler_id = %s", (labeler_id,)
            )
            remaining_labels = cursor.fetchone()[0]

            labelers_removed = 0
            if remaining_labels == 0:
                cursor.execute("DELETE FROM labeler WHERE id = %s", (labeler_id,))
                labelers_removed = cursor.rowcount
                logger.info(f"Removed labeler {labeler_id} (no remaining data)")
            else:
                logger.info(
                    f"Kept labeler {labeler_id} ({remaining_labels} labels remaining)"
                )

            conn.commit()

            return {
                "queries_removed": queries_removed,
                "examples_removed": examples_removed,
                "labels_removed": labels_removed,
                "labelers_removed": labelers_removed,
            }


def reset_mlflow_status(run_id: str) -> None:
    """Reset MLflow run status from migrated to pending_approval."""
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "data_status", "pending_approval")
    client.set_tag(run_id, "rollback_timestamp", datetime.now().isoformat())
    logger.info(f"Reset run {run_id} status to pending_approval")


def rollback_run(run_id: str, dry_run: bool = True) -> dict:
    """Rollback a single migrated MLflow run."""

    try:
        # Get run data
        run_data = get_migrated_run(run_id)

        data_gen_hash = run_data["data_gen_hash"]
        mlflow_run_id = run_data["mlflow_run_id"]
        generation_approach = run_data["generation_approach"]

        logger.info(f"Rolling back run: {run_id} ({run_data['run_name']})")
        logger.info(f"  Approach: {generation_approach}, Git hash: {data_gen_hash}")

        # Remove database records
        removal_stats = remove_run_data(
            data_gen_hash, mlflow_run_id, generation_approach, dry_run
        )

        if not dry_run:
            # Reset MLflow status
            reset_mlflow_status(run_id)

        return {
            "run_id": run_id,
            "status": "dry_run" if dry_run else "success",
            **removal_stats,
        }

    except Exception as e:
        logger.error(f"Failed to rollback run {run_id}: {e}")
        return {
            "run_id": run_id,
            "status": "error",
            "error": str(e),
            "queries_removed": 0,
            "examples_removed": 0,
            "labels_removed": 0,
            "labelers_removed": 0,
        }


def main():
    """Main rollback execution."""
    parser = argparse.ArgumentParser(description="Rollback synthetic data migration")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to rollback")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without making changes"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm rollback (required for actual rollback)",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.confirm:
        parser.error(
            "Must use --confirm flag for actual rollback (or --dry-run to preview)"
        )

    try:
        # Setup MLflow
        setup_mlflow()

        logger.info("=" * 60)
        logger.info("SYNTHETIC DATA ROLLBACK STARTED")
        logger.info("=" * 60)

        # Rollback the run
        result = rollback_run(args.run_id, dry_run=args.dry_run)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("ROLLBACK SUMMARY")
        logger.info("=" * 60)

        if result["status"] == "dry_run":
            logger.info(f"[DRY RUN] Would rollback run {result['run_id']}")
            logger.info(
                f"  Would remove: {result['queries_removed']} queries, "
                f"{result['examples_removed']} examples, "
                f"{result['labels_removed']} labels, "
                f"{result['labelers_removed']} labelers"
            )
        elif result["status"] == "success":
            logger.info(f"Successfully rolled back run {result['run_id']}")
            logger.info(
                f"  Removed: {result['queries_removed']} queries, "
                f"{result['examples_removed']} examples, "
                f"{result['labels_removed']} labels, "
                f"{result['labelers_removed']} labelers"
            )
        else:
            logger.error(
                f"Rollback failed for run {result['run_id']}: {result['error']}"
            )

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise


if __name__ == "__main__":
    main()
