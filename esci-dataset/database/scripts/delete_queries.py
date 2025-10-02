#!/usr/bin/env python3
"""
Delete Queries Script

This script deletes queries and their associated data from the database by:
1. Finding queries by ID(s) or content pattern
2. Deleting all labels for examples linked to these queries
3. Deleting all examples linked to these queries
4. Deleting the queries themselves

Usage:
    # Delete by specific IDs
    python delete_queries.py --ids 1001 1002 1003 --dry-run
    python delete_queries.py --ids 1001 1002 --confirm

    # Delete by content pattern (case-insensitive, partial match)
    python delete_queries.py --pattern "pizza" --dry-run
    python delete_queries.py --pattern "biryani" --confirm

    # Delete by exact content match (case-insensitive)
    python delete_queries.py --pattern "paneer tikka pizza" --exact --dry-run
    python delete_queries.py --pattern "chicken biryani" --exact --confirm

    # Delete by pattern list (e.g., vague/generic queries)
    python delete_queries.py --pattern-list "food,meal,dish,something" --dry-run
"""

import argparse
import logging
import os
import sys

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
        logging.FileHandler("delete_queries.log"),
    ],
)
logger = logging.getLogger(__name__)


def find_queries_by_ids(query_ids: list[str]) -> list[tuple[int, str]]:
    """Find queries by their IDs.

    Args:
        query_ids: List of query IDs to find (as strings)

    Returns:
        List of tuples (query_id, query_content)
    """
    # Convert string IDs to integers
    int_ids = [int(id_str) for id_str in query_ids]

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, query_content
                FROM query
                WHERE id = ANY(%s)
                ORDER BY id
                """,
                (int_ids,),
            )
            results = cursor.fetchall()
            return [(row[0], row[1]) for row in results]


def find_queries_by_pattern(pattern: str, exact: bool = False) -> list[tuple[int, str]]:
    """Find queries by content pattern (case-insensitive).

    Args:
        pattern: Pattern to match in query content
        exact: If True, match exactly; if False, match substring (default)

    Returns:
        List of tuples (query_id, query_content)
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            if exact:
                # Exact match (case-insensitive)
                cursor.execute(
                    """
                    SELECT id, query_content
                    FROM query
                    WHERE LOWER(query_content) = LOWER(%s)
                    ORDER BY query_content
                    """,
                    (pattern,),
                )
            else:
                # Partial match (case-insensitive substring)
                cursor.execute(
                    """
                    SELECT id, query_content
                    FROM query
                    WHERE LOWER(query_content) LIKE LOWER(%s)
                    ORDER BY query_content
                    """,
                    (f"%{pattern}%",),
                )
            results = cursor.fetchall()
            return [(row[0], row[1]) for row in results]


def find_queries_by_pattern_list(patterns: list[str]) -> list[tuple[int, str]]:
    """Find queries matching any of the patterns in the list.

    Args:
        patterns: List of patterns to match

    Returns:
        List of tuples (query_id, query_content)
    """
    queries = set()
    for pattern in patterns:
        results = find_queries_by_pattern(pattern.strip())
        queries.update(results)
    return sorted(list(queries), key=lambda x: x[1])


def get_deletion_stats(query_id: int) -> dict:
    """Get statistics about what will be deleted for a query.

    Args:
        query_id: Query ID to analyze

    Returns:
        Dictionary with counts of examples and labels
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Count examples
            cursor.execute(
                "SELECT COUNT(*) FROM example WHERE query_id = %s",
                (query_id,),
            )
            example_count = cursor.fetchone()[0]

            # Count labels
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM label l
                JOIN example e ON l.example_id = e.id
                WHERE e.query_id = %s
                """,
                (query_id,),
            )
            label_count = cursor.fetchone()[0]

            return {
                "examples": example_count,
                "labels": label_count,
            }


def delete_query(query_id: int, dry_run: bool = True) -> dict:
    """Delete a query and all associated data.

    Args:
        query_id: Query ID to delete
        dry_run: If True, only simulate deletion without committing

    Returns:
        Dictionary with deletion statistics
    """
    stats = get_deletion_stats(query_id)

    if dry_run:
        logger.info(
            f"[DRY RUN] Would delete query {query_id}: "
            f"{stats['examples']} examples, {stats['labels']} labels"
        )
        return stats

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get example IDs
                cursor.execute(
                    "SELECT id FROM example WHERE query_id = %s",
                    (query_id,),
                )
                example_ids = [row[0] for row in cursor.fetchall()]

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
                cursor.execute("DELETE FROM example WHERE query_id = %s", (query_id,))
                examples_deleted = cursor.rowcount

                # Delete the query
                cursor.execute("DELETE FROM query WHERE id = %s", (query_id,))

                conn.commit()

                logger.info(
                    f"Deleted query {query_id}: "
                    f"{labels_deleted} labels, {examples_deleted} examples"
                )

                return {
                    "examples": examples_deleted,
                    "labels": labels_deleted,
                }

    except Exception as e:
        logger.error(f"Error deleting query {query_id}: {e}")
        raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Delete queries and their associated data from the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ids", nargs="+", help="Query IDs to delete (space-separated)"
    )
    input_group.add_argument(
        "--pattern",
        help="Content pattern to match (case-insensitive, uses LIKE by default)",
    )
    input_group.add_argument(
        "--pattern-list",
        help="Comma-separated patterns to match (e.g., 'food,meal,dish')",
    )

    # Matching options
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact content match instead of substring match (only with --pattern)",
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
        logger.error("Error: Either --dry-run or --confirm flag is required to proceed")
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

    # Find queries based on input method
    queries = []
    if args.ids:
        logger.info(f"Finding queries by IDs: {args.ids}")
        queries = find_queries_by_ids(args.ids)
    elif args.pattern:
        match_type = "exact" if args.exact else "substring"
        logger.info(
            f"Finding queries matching pattern: '{args.pattern}' ({match_type} match)"
        )
        queries = find_queries_by_pattern(args.pattern, exact=args.exact)
    elif args.pattern_list:
        patterns = [p.strip() for p in args.pattern_list.split(",")]
        logger.info(f"Finding queries matching patterns: {patterns}")
        queries = find_queries_by_pattern_list(patterns)

    if not queries:
        logger.warning("No queries found matching the criteria")
        sys.exit(0)

    logger.info(f"Found {len(queries)} query(ies) to delete:")
    for query_id, query_content in queries:
        logger.info(f"  - {query_id}: {query_content}")

    # Get deletion preview
    logger.info("\nAnalyzing deletion impact...")
    total_stats = {"examples": 0, "labels": 0}

    for query_id, query_content in queries:
        stats = get_deletion_stats(query_id)
        logger.info(
            f"  {query_id} ({query_content}): "
            f"{stats['examples']} examples, {stats['labels']} labels"
        )
        for key in total_stats:
            total_stats[key] += stats[key]

    logger.info("\nTotal impact:")
    logger.info(f"  Queries: {len(queries)}")
    logger.info(f"  Examples: {total_stats['examples']}")
    logger.info(f"  Labels: {total_stats['labels']}")

    if args.dry_run:
        logger.info("\n[DRY RUN] No changes made. Use --confirm to delete.")
        sys.exit(0)

    # Confirm deletion
    logger.warning("\n⚠️  DELETION WILL BE PERMANENT AND CANNOT BE UNDONE!")
    confirmation = input(
        f"Type 'DELETE' to confirm deletion of {len(queries)} query(ies): "
    )

    if confirmation != "DELETE":
        logger.info("Deletion cancelled.")
        sys.exit(0)

    # Perform deletion
    logger.info("\nDeleting queries...")
    success_count = 0
    error_count = 0

    for query_id, query_content in queries:
        try:
            delete_query(query_id, dry_run=False)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {query_id} ({query_content}): {e}")
            error_count += 1

    logger.info(f"\nDeletion complete: {success_count} succeeded, {error_count} failed")

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
