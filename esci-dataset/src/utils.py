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
