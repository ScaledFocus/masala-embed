#!/usr/bin/env python3
"""
Quick script to clear consumable table without pandas dependency
"""

import argparse
import sys

from database.utils.db_utils import drop_all_records


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Clear all records from a database table"
    )
    parser.add_argument("--table", required=True, help="Name of the table to clear")
    parser.add_argument(
        "--confirm", action="store_true", help="Confirm deletion (required for safety)"
    )

    args = parser.parse_args()

    if not args.confirm:
        print("Error: --confirm flag is required to proceed with deletion")
        print("This is a safety measure to prevent accidental data loss")
        sys.exit(1)

    print(f"Clearing {args.table} table...")
    result = drop_all_records(args.table, confirm=args.confirm)

    print("Operation completed!")
    print(f"Table: {result['table_name']}")
    print(f"Initial count: {result['initial_count']:,} records")
    print(f"Deleted count: {result['deleted_count']:,} records")
    print(f"Final count: {result['final_count']:,} records")
    print(f"Sequences reset: {result['sequences_reset']}")


if __name__ == "__main__":
    main()
