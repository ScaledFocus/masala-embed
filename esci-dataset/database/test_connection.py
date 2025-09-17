"""Test database connection and query labeler table."""

import os

import psycopg2
from dotenv import load_dotenv


def main():
    """Test database connection."""
    load_dotenv()

    # Load environment variables from .env
    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host")
    port = os.getenv("port")
    dbname = os.getenv("dbname")

    # Connect to the database
    try:
        connection = psycopg2.connect(
            user=user, password=password, host=host, port=port, dbname=dbname
        )
        print("Connection successful!")

        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

        # Example queries (read-only, no commit needed)
        cursor.execute("SELECT NOW();")
        result = cursor.fetchone()
        cursor.execute("SELECT * FROM public.labeler;")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

        print("Current Time:", result)

        # Close the cursor and connection
        cursor.close()
        connection.close()
        print("Connection closed.")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Failed to connect: {e}")


if __name__ == "__main__":
    main()
