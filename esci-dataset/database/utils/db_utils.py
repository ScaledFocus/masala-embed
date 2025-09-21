import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Create database connection using environment variables."""
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
    )


def get_db_info():
    """Get database connection info and table counts."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user, version();")
            db_info = cursor.fetchone()

            cursor.execute("""
                SELECT table_name,
                (xpath('/row/c/text()',
                query_to_xml('SELECT count(*) as c
                FROM ' || table_name, false, true, '')))[1]::text::int as row_count
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()

    return {
        "database": db_info[0],
        "user": db_info[1],
        "version": db_info[2],
        "tables": {table[0]: table[1] for table in tables},
    }


def get_table(table_name, limit=None, columns="*", where=None, order_by=None):
    """Read table data into pandas DataFrame."""
    query = f"SELECT {columns} FROM {table_name}"

    if where:
        query += f" WHERE {where}"
    if order_by:
        query += f" ORDER BY {order_by}"
    if limit:
        query += f" LIMIT {limit}"

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)

    return df


def execute_query(query, params=None):
    """Execute a SQL query and return results as DataFrame."""
    with get_db_connection() as conn:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
    return df


def drop_all_records(table_name: str, confirm: bool = False) -> dict[str, int]:
    """
    Drop all records from a specified table.

    Args:
        table_name: Name of the table to clear
        confirm: Safety flag - must be True to execute

    Returns:
        Dictionary with operation statistics

    Example:
        # Clear consumable table
        result = drop_all_records("consumable", confirm=True)
        print(f"Deleted {result['deleted_count']} records")
    """
    if not confirm:
        raise ValueError("Safety check: confirm=True required to drop all records")

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Get count before deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            initial_count = cursor.fetchone()[0]

            # Delete all records
            cursor.execute(f"DELETE FROM {table_name}")
            deleted_count = cursor.rowcount

            # Reset auto-increment sequence (if applicable)
            cursor.execute(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND column_default LIKE 'nextval%%'
            """, (table_name,))

            auto_increment_cols = cursor.fetchall()
            if auto_increment_cols:
                # Reset sequence for auto-increment columns
                for col in auto_increment_cols:
                    sequence_name = f"{table_name}_{col[0]}_seq"
                    cursor.execute(f"ALTER SEQUENCE {sequence_name} RESTART WITH 1")

            conn.commit()

            # Verify deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            final_count = cursor.fetchone()[0]

    return {
        "table_name": table_name,
        "initial_count": initial_count,
        "deleted_count": deleted_count,
        "final_count": final_count,
        "sequences_reset": len(auto_increment_cols)
    }