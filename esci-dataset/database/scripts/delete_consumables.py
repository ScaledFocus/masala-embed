#!/usr/bin/env python3
"""
Delete Consumables Script

This script deletes consumables and their associated data from the database by:
1. Finding consumables by ID(s) or name pattern
2. Deleting all labels for examples linked to these consumables
3. Deleting all examples linked to these consumables
4. Deleting orphaned queries (queries with no remaining examples)
5. Deleting the consumables themselves

Usage:
    # Delete by specific IDs
    python delete_consumables.py --ids C001 C002 C003 --dry-run
    python delete_consumables.py --ids C001 C002 --confirm

    # Delete by name pattern (case-insensitive, partial match)
    python delete_consumables.py --pattern "mixed platter" --dry-run
    python delete_consumables.py --pattern "assorted" --confirm

    # Delete by exact name match (case-insensitive)
    python delete_consumables.py --pattern "Mixed Platter" --exact --dry-run
    python delete_consumables.py --pattern "Assorted Snacks" --exact --confirm

    # Delete vague/generic items (e.g., contains "mixed", "assorted", "platter")
    python delete_consumables.py --pattern-list "mixed,assorted,platter,combo" --dry-run
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

# Add project paths
project_root = os.getenv("root_folder")
if project_root:
    sys.path.append(os.path.join(project_root, "esci-dataset"))

from database.utils.db_utils import get_db_connection  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("delete_consumables.log"),
    ],
)
logger = logging.getLogger(__name__)


def find_consumables_by_ids(consumable_ids: List[str]) -> List[Tuple[str, str]]:
    """Find consumables by their IDs.

    Args:
        consumable_ids: List of consumable IDs to find

    Returns:
        List of tuples (consumable_id, consumable_name)
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, consumable_name
                FROM consumable
                WHERE id = ANY(%s)
                ORDER BY id
                """,
                (consumable_ids,),
            )
            results = cursor.fetchall()
            return [(row[0], row[1]) for row in results]


def find_consumables_by_pattern(pattern: str, exact: bool = False) -> List[Tuple[str, str]]:
    """Find consumables by name pattern (case-insensitive).

    Args:
        pattern: Pattern to match in consumable names
        exact: If True, match exactly; if False, match substring (default)

    Returns:
        List of tuples (consumable_id, consumable_name)
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            if exact:
                # Exact match (case-insensitive)
                cursor.execute(
                    """
                    SELECT id, consumable_name
                    FROM consumable
                    WHERE LOWER(consumable_name) = LOWER(%s)
                    ORDER BY consumable_name
                    """,
                    (pattern,),
                )
            else:
                # Partial match (case-insensitive substring)
                cursor.execute(
                    """
                    SELECT id, consumable_name
                    FROM consumable
                    WHERE LOWER(consumable_name) LIKE LOWER(%s)
                    ORDER BY consumable_name
                    """,
                    (f"%{pattern}%",),
                )
            results = cursor.fetchall()
            return [(row[0], row[1]) for row in results]


def find_consumables_by_pattern_list(patterns: List[str]) -> List[Tuple[str, str]]:
    """Find consumables matching any of the patterns in the list.

    Args:
        patterns: List of patterns to match

    Returns:
        List of tuples (consumable_id, consumable_name)
    """
    consumables = set()
    for pattern in patterns:
        results = find_consumables_by_pattern(pattern.strip())
        consumables.update(results)
    return sorted(list(consumables), key=lambda x: x[1])


def get_deletion_stats(consumable_id: str) -> dict:
    """Get statistics about what will be deleted for a consumable.

    Args:
        consumable_id: Consumable ID to analyze

    Returns:
        Dictionary with counts of examples, labels, and queries
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Count examples
            cursor.execute(
                "SELECT COUNT(*) FROM example WHERE consumable_id = %s",
                (consumable_id,),
            )
            example_count = cursor.fetchone()[0]

            # Get query IDs
            cursor.execute(
                "SELECT DISTINCT query_id FROM example WHERE consumable_id = %s",
                (consumable_id,),
            )
            query_ids = [row[0] for row in cursor.fetchall()]

            # Count labels
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM label l
                JOIN example e ON l.example_id = e.id
                WHERE e.consumable_id = %s
                """,
                (consumable_id,),
            )
            label_count = cursor.fetchone()[0]

            # Count orphaned queries (queries that would have no examples left)
            orphaned_query_count = 0
            for query_id in query_ids:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM example
                    WHERE query_id = %s AND consumable_id != %s
                    """,
                    (query_id, consumable_id),
                )
                other_examples = cursor.fetchone()[0]
                if other_examples == 0:
                    orphaned_query_count += 1

            return {
                "examples": example_count,
                "labels": label_count,
                "queries": len(query_ids),
                "orphaned_queries": orphaned_query_count,
            }


def delete_consumable(consumable_id: str, dry_run: bool = True) -> dict:
    """Delete a consumable and all associated data.

    Args:
        consumable_id: Consumable ID to delete
        dry_run: If True, only simulate deletion without committing

    Returns:
        Dictionary with deletion statistics
    """
    stats = get_deletion_stats(consumable_id)

    if dry_run:
        logger.info(
            f"[DRY RUN] Would delete consumable {consumable_id}: "
            f"{stats['examples']} examples, {stats['labels']} labels, "
            f"{stats['orphaned_queries']} orphaned queries"
        )
        return stats

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get example IDs and query IDs
                cursor.execute(
                    "SELECT id, query_id FROM example WHERE consumable_id = %s",
                    (consumable_id,),
                )
                examples = cursor.fetchall()
                example_ids = [ex[0] for ex in examples]
                query_ids = [ex[1] for ex in examples]

                # Delete labels
                if example_ids:
                    cursor.execute(
                        "DELETE FROM label WHERE example_id = ANY(%s)",
                        (example_ids,),
                    )
                    labels_deleted = cursor.rowcount
                else:
                    labels_deleted = 0

                # Delete examples
                cursor.execute(
                    "DELETE FROM example WHERE consumable_id = %s", (consumable_id,)
                )
                examples_deleted = cursor.rowcount

                # Delete orphaned queries
                queries_deleted = 0
                for query_id in set(query_ids):
                    cursor.execute(
                        "SELECT COUNT(*) FROM example WHERE query_id = %s", (query_id,)
                    )
                    remaining_examples = cursor.fetchone()[0]
                    if remaining_examples == 0:
                        cursor.execute("DELETE FROM query WHERE id = %s", (query_id,))
                        queries_deleted += 1

                # Delete the consumable
                cursor.execute("DELETE FROM consumable WHERE id = %s", (consumable_id,))

                conn.commit()

                logger.info(
                    f"Deleted consumable {consumable_id}: "
                    f"{labels_deleted} labels, {examples_deleted} examples, "
                    f"{queries_deleted} orphaned queries"
                )

                return {
                    "examples": examples_deleted,
                    "labels": labels_deleted,
                    "queries": len(set(query_ids)),
                    "orphaned_queries": queries_deleted,
                }

    except Exception as e:
        logger.error(f"Error deleting consumable {consumable_id}: {e}")
        raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Delete consumables and their associated data from the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ids", nargs="+", help="Consumable IDs to delete (space-separated)"
    )
    input_group.add_argument(
        "--pattern", help="Name pattern to match (case-insensitive, uses LIKE by default)"
    )
    input_group.add_argument(
        "--pattern-list",
        help="Comma-separated list of patterns to match (e.g., 'mixed,assorted,platter')",
    )

    # Matching options
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact name match instead of substring match (only with --pattern)",
    )

    # Action options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without making changes",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Confirm deletion (required for safety)"
    )

    args = parser.parse_args()

    # Validate action flags
    if not args.dry_run and not args.confirm:
        logger.error(
            "Error: Either --dry-run or --confirm flag is required to proceed"
        )
        logger.error("Use --dry-run to preview changes without making them")
        logger.error("Use --confirm to actually delete the data")
        sys.exit(1)

    if args.dry_run and args.confirm:
        logger.error("Error: Cannot use both --dry-run and --confirm")
        sys.exit(1)

    # Validate --exact flag usage
    if args.exact and not args.pattern:
        logger.error("Error: --exact can only be used with --pattern")
        sys.exit(1)

    # Find consumables based on input method
    consumables = []
    if args.ids:
        logger.info(f"Finding consumables by IDs: {args.ids}")
        consumables = find_consumables_by_ids(args.ids)
    elif args.pattern:
        match_type = "exact" if args.exact else "substring"
        logger.info(f"Finding consumables matching pattern: '{args.pattern}' ({match_type} match)")
        consumables = find_consumables_by_pattern(args.pattern, exact=args.exact)
    elif args.pattern_list:
        patterns = [p.strip() for p in args.pattern_list.split(",")]
        logger.info(f"Finding consumables matching patterns: {patterns}")
        consumables = find_consumables_by_pattern_list(patterns)

    if not consumables:
        logger.warning("No consumables found matching the criteria")
        sys.exit(0)

    logger.info(f"Found {len(consumables)} consumable(s) to delete:")
    for consumable_id, consumable_name in consumables:
        logger.info(f"  - {consumable_id}: {consumable_name}")

    # Get deletion preview
    logger.info("\nAnalyzing deletion impact...")
    total_stats = {"examples": 0, "labels": 0, "queries": 0, "orphaned_queries": 0}

    for consumable_id, consumable_name in consumables:
        stats = get_deletion_stats(consumable_id)
        logger.info(
            f"  {consumable_id} ({consumable_name}): "
            f"{stats['examples']} examples, {stats['labels']} labels, "
            f"{stats['orphaned_queries']}/{stats['queries']} orphaned queries"
        )
        for key in total_stats:
            total_stats[key] += stats[key]

    logger.info("\nTotal impact:")
    logger.info(f"  Consumables: {len(consumables)}")
    logger.info(f"  Examples: {total_stats['examples']}")
    logger.info(f"  Labels: {total_stats['labels']}")
    logger.info(
        f"  Orphaned queries: {total_stats['orphaned_queries']}/{total_stats['queries']}"
    )

    if args.dry_run:
        logger.info("\n[DRY RUN] No changes made. Use --confirm to delete.")
        sys.exit(0)

    # Confirm deletion
    logger.warning("\n⚠️  DELETION WILL BE PERMANENT AND CANNOT BE UNDONE!")
    confirmation = input(f"Type 'DELETE' to confirm deletion of {len(consumables)} consumable(s): ")

    if confirmation != "DELETE":
        logger.info("Deletion cancelled.")
        sys.exit(0)

    # Perform deletion
    logger.info("\nDeleting consumables...")
    success_count = 0
    error_count = 0

    for consumable_id, consumable_name in consumables:
        try:
            delete_consumable(consumable_id, dry_run=False)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {consumable_id} ({consumable_name}): {e}")
            error_count += 1

    logger.info(
        f"\nDeletion complete: {success_count} succeeded, {error_count} failed"
    )

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()