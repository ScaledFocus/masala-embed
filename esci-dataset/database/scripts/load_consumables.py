#!/usr/bin/env python3
"""
Load Consumables Pipeline

Complete pipeline to process MM-Food-100K dataset and load into consumable table:
1. Remove null dish_name rows
2. Hybrid deduplication (within dish groups)
3. Transform columns for database schema
4. Batch insert into consumable table

Usage:
    python load_consumables.py --input MM-Food-100K.csv --threshold 0.9 --batch-size 500
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

# Add paths for imports using absolute paths from .env
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from tqdm import tqdm

load_dotenv()

project_root = os.getenv("root_folder")
if project_root:
    sys.path.append(os.path.join(project_root, "esci-dataset", "src"))
    sys.path.append(project_root)

from database.utils.db_utils import get_db_connection  # noqa: E402
from src.preprocessing import (  # noqa: E402
    hybrid_deduplicate_dataset,
    prepare_consumable_data,
    remove_null_rows,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("load_consumables.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Load consumables into database")
    parser.add_argument("--input", default="MM-Food-100K.csv", help="Input CSV file")
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="Deduplication threshold"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Batch size for insertion"
    )
    parser.add_argument("--data-folder", help="Data folder path (overrides env var)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without inserting to database"
    )

    args = parser.parse_args()

    try:
        # Step 1: Load and clean data
        logger.info("=" * 60)
        logger.info("CONSUMABLES LOADING PIPELINE STARTED")
        logger.info("=" * 60)

        df_cleaned = load_and_clean_data(args.input, args.data_folder)
        logger.info(f"After cleaning: {len(df_cleaned):,} records")

        # Step 2: Hybrid deduplication
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: HYBRID DEDUPLICATION")
        logger.info("=" * 40)

        df_deduplicated = run_hybrid_deduplication(
            df_cleaned, args.threshold, args.data_folder
        )
        logger.info(f"After deduplication: {len(df_deduplicated):,} records")

        # Step 3: Transform for database
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: TRANSFORM FOR DATABASE")
        logger.info("=" * 40)

        df_prepared = prepare_consumable_data(df_deduplicated)
        logger.info(f"Prepared for insertion: {len(df_prepared):,} records")

        # Step 4: Database insertion
        if not args.dry_run:
            logger.info("\n" + "=" * 40)
            logger.info("STEP 4: DATABASE INSERTION")
            logger.info("=" * 40)

            insertion_stats = insert_consumables_batch(df_prepared, args.batch_size)

            # Final results
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Records processed: {len(df_cleaned):,}")
            logger.info(f"Records deduplicated: {len(df_deduplicated):,}")
            logger.info(f"Records inserted: {insertion_stats['total_inserted']:,}")
            logger.info(
                f"Success rate: "
                f"{insertion_stats['total_inserted'] / len(df_prepared) * 100:.1f}%"
            )
        else:
            logger.info("\n" + "=" * 40)
            logger.info("DRY RUN COMPLETED")
            logger.info("=" * 40)
            logger.info(f"Would insert {len(df_prepared):,} records")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def load_and_clean_data(input_file: str, data_folder: str = None) -> pd.DataFrame:
    """Load CSV data and remove null dish_name rows."""
    logger.info("STEP 1: LOAD AND CLEAN DATA")
    logger.info("=" * 40)

    # Determine data folder
    if data_folder is None:
        from dotenv import load_dotenv

        load_dotenv()
        data_folder = os.environ.get("data_folder")
        if not data_folder:
            raise ValueError(
                "data_folder not provided and not found in environment variables"
            )

    # Load data
    input_path = os.path.join(data_folder, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} records from {input_file}")

    # Remove null dish_name rows
    logger.info("Removing rows with null/empty dish_name...")
    df_cleaned = remove_null_rows(df, ["dish_name"])

    return df_cleaned


def run_hybrid_deduplication(
    df: pd.DataFrame, threshold: float, data_folder: str = None
) -> pd.DataFrame:
    """Run hybrid deduplication on the dataset."""
    # Create temporary file for deduplication
    temp_input = "temp_cleaned.csv"
    temp_output = "temp_deduplicated.csv"

    if data_folder is None:
        from dotenv import load_dotenv

        load_dotenv()
        data_folder = os.environ.get("data_folder")

    temp_input_path = os.path.join(data_folder, temp_input)
    temp_output_path = os.path.join(data_folder, temp_output)

    try:
        # Save cleaned data to temp file
        df.to_csv(temp_input_path, index=False)
        logger.info(f"Saved cleaned data to: {temp_input_path}")

        # Run hybrid deduplication
        logger.info(f"Running hybrid deduplication with threshold: {threshold}")
        df_deduplicated = hybrid_deduplicate_dataset(
            input_file=temp_input,
            output_file=temp_output,
            threshold=threshold,
            data_folder=data_folder,
        )

        logger.info(f"Deduplication completed. Output saved to: {temp_output_path}")
        return df_deduplicated

    finally:
        # Clean up temp files
        for temp_file in [temp_input_path, temp_output_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Removed temp file: {temp_file}")


def insert_consumables_batch(df: pd.DataFrame, batch_size: int = 500) -> dict[str, int]:
    """Insert consumables data in batches with progress tracking."""

    # Prepare SQL statement (no ID column - auto-increment)
    sql = """
    INSERT INTO consumable (
        image_url, consumable_name, consumable_type,
        consumable_ingredients, consumable_portion_size,
        consumable_nutritional_profile, consumable_cooking_method
    ) VALUES %s
    """

    stats = {
        "total_records": len(df),
        "successful_batches": 0,
        "failed_batches": 0,
        "total_inserted": 0,
        "errors": [],
    }

    logger.info(f"Starting batch insertion of {len(df):,} records")
    logger.info(f"Batch size: {batch_size}")

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Get initial count
            cursor.execute("SELECT COUNT(*) FROM consumable")
            initial_count = cursor.fetchone()[0]
            logger.info(f"Initial database count: {initial_count:,} records")

            # Process data in batches with progress bar
            pbar = tqdm(total=len(df), desc="Inserting records", unit="records")

            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i : i + batch_size]

                try:
                    # Prepare batch data (no ID - auto-increment)
                    batch_data = []
                    for _, row in batch_df.iterrows():
                        # Convert nutritional profile to JSON string for PostgreSQL
                        nutrition = row.get("consumable_nutritional_profile")
                        if nutrition is not None:
                            nutrition = (
                                json.dumps(nutrition)
                                if not isinstance(nutrition, str)
                                else nutrition
                            )

                        batch_data.append(
                            (
                                row.get("image_url"),
                                row.get("consumable_name"),
                                row.get("consumable_type"),
                                row.get("consumable_ingredients"),
                                row.get("consumable_portion_size"),
                                nutrition,
                                row.get("consumable_cooking_method"),
                            )
                        )

                    # Execute batch insert
                    execute_values(
                        cursor, sql, batch_data, template=None, page_size=batch_size
                    )

                    stats["successful_batches"] += 1
                    stats["total_inserted"] += len(batch_data)

                    pbar.update(len(batch_data))

                except Exception as e:
                    stats["failed_batches"] += 1
                    error_msg = f"Batch {i // batch_size + 1} failed: {str(e)}"
                    stats["errors"].append(error_msg)
                    logger.error(error_msg)
                    pbar.update(len(batch_data))
                    # Continue with next batch instead of breaking
                    continue

            pbar.close()

            # Commit all changes
            conn.commit()

            # Get final count
            cursor.execute("SELECT COUNT(*) FROM consumable")
            final_count = cursor.fetchone()[0]

            logger.info(
                f"Insertion completed: {stats['total_inserted']:,}/"
                f"{stats['total_records']:,} records"
            )
            logger.info(
                f"Database count: {initial_count:,} → {final_count:,} "
                f"(+{final_count - initial_count:,})"
            )

            if stats["failed_batches"] > 0:
                logger.warning(f"{stats['failed_batches']} batches failed")
                for error in stats["errors"][:3]:  # Show first 3 errors
                    logger.warning(f"  - {error}")
                if len(stats["errors"]) > 3:
                    logger.warning(f"  ... and {len(stats['errors']) - 3} more errors")

    return stats


if __name__ == "__main__":
    main()
