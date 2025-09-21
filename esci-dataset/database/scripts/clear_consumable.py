#!/usr/bin/env python3
"""
Quick script to clear consumable table without pandas dependency
"""
import os
import sys
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

def drop_all_records(table_name: str, confirm: bool = False):
    """Drop all records from a specified table."""
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

if __name__ == "__main__":
    print("Clearing consumable table...")
    result = drop_all_records('consumable', confirm=True)
    
    print("Operation completed!")
    print(f"Table: {result['table_name']}")
    print(f"Initial count: {result['initial_count']:,} records")
    print(f"Deleted count: {result['deleted_count']:,} records")
    print(f"Final count: {result['final_count']:,} records")
    print(f"Sequences reset: {result['sequences_reset']}")