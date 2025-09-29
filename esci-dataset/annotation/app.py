#!/usr/bin/env python3
"""
Flask-based ESCI Annotation Tool
Supports both CSV files and database storage

Usage:
    python app.py queries_I_batch100_limit200_v2_20250925_112245.csv
    python app.py --database
"""

import argparse
import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request

# Import database utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "database", "utils"))
from db_utils import execute_query, get_db_connection

app = Flask(__name__)

# Global variables
csv_file_path = None
df = None
use_database = False
mlflow_run_id = None
labeler_id = None


@app.route("/")
def index():
    return render_template("annotate.html")


@app.route("/api/data")
def get_data():
    """Get all annotation data, optionally filtered for review mode"""
    global df
    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    # Check if review mode is requested
    review_mode = request.args.get("review_mode", "false").lower() == "true"

    # Filter data based on review mode
    if review_mode:
        if use_database:
            # Database mode: filter for records with human labels
            filtered_df = df[
                df["human_esci_label"].notna() & (df["human_esci_label"] != "")
            ]
        else:
            # CSV mode: filter for records with esci_label
            filtered_df = df[df["esci_label"].notna() & (df["esci_label"] != "")]

        data_to_return = filtered_df.to_dict("records")
        total_count = len(filtered_df)
    else:
        data_to_return = df.to_dict("records")
        total_count = len(df)

    # Get filename based on mode
    if use_database:
        filename = f"Database: Run {mlflow_run_id}"
    else:
        filename = os.path.basename(csv_file_path)

    return jsonify(
        {
            "data": data_to_return,
            "total": total_count,
            "filename": filename,
            "review_mode": review_mode,
            "total_records": len(df),  # Always include total record count
        }
    )


@app.route("/api/update", methods=["POST"])
def update_label():
    """Update ESCI label for a specific row"""
    global df, csv_file_path, use_database

    data = request.json
    index = data.get("index")
    example_id = data.get("example_id")
    label = data.get("label")

    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    if label not in ["E", "S", "C", "I"]:
        return jsonify({"error": "Invalid label"}), 400

    if use_database:
        # Database mode: require example_id for proper record identification
        if not example_id:
            return jsonify({"error": "example_id required for database mode"}), 400

        if not save_label_to_database_by_id(example_id, label):
            return jsonify({"error": "Failed to save to database"}), 500
        # Update DataFrame for the matching example_id
        mask = df["example_id"] == example_id
        if mask.any():
            df.loc[mask, "human_esci_label"] = label
        message = f"Updated to label {label}"
    else:
        # CSV mode: use index-based approach
        if index < 0 or index >= len(df):
            return jsonify({"error": "Invalid index"}), 400

        # Update the DataFrame and save CSV
        df.at[index, "esci_label"] = label
        df.to_csv(csv_file_path, index=False)
        message = f"Updated row {index} to label {label}"

    return jsonify({"success": True, "message": message})


@app.route("/api/update-query", methods=["POST"])
def update_query():
    """Update query text for a specific record"""
    global df, use_database

    data = request.json
    index = data.get("index")
    example_id = data.get("example_id")
    new_query = data.get("query", "").strip()

    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    if not new_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if use_database:
        # Database mode: require example_id for proper record identification
        if not example_id:
            return jsonify({"error": "example_id required for database mode"}), 400

        if not update_query_in_database_by_id(example_id, new_query):
            return jsonify({"error": "Failed to update query in database"}), 500
    else:
        # CSV mode: use index-based approach
        if index < 0 or index >= len(df):
            return jsonify({"error": "Invalid index"}), 400

        df.at[index, "query"] = new_query
        df.to_csv(csv_file_path, index=False)

    return jsonify({"success": True, "message": "Query updated successfully"})


@app.route("/api/delete-example", methods=["POST"])
def delete_example():
    """Delete an entire example and all its labels"""
    global df, use_database

    data = request.json
    index = data.get("index")
    example_id = data.get("example_id")

    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    if use_database:
        # Database mode: require example_id for proper record identification
        if not example_id:
            return jsonify({"error": "example_id required for database mode"}), 400

        success, labels_deleted = delete_example_from_database_by_id(example_id)
        if not success:
            return jsonify({"error": "Failed to delete example from database"}), 500
        message = f"Deleted example and {labels_deleted} labels"
    else:
        # CSV mode: use index-based approach
        if index < 0 or index >= len(df):
            return jsonify({"error": "Invalid index"}), 400

        df.drop(df.index[index], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(csv_file_path, index=False)
        message = "Deleted example from CSV"

    return jsonify({"success": True, "message": message})


@app.route("/api/delete-label", methods=["POST"])
def delete_label():
    """Delete only the current user's label for an example"""
    global df, use_database

    data = request.json
    index = data.get("index")
    example_id = data.get("example_id")  # New: get example_id if provided

    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    if use_database:
        # Database mode: require example_id for proper record identification
        if not example_id:
            return jsonify({"error": "example_id required for database mode"}), 400

        if not delete_label_from_database_by_id(example_id):
            return jsonify({"error": "Failed to delete label from database"}), 500
        # Update DataFrame for the matching example_id
        mask = df["example_id"] == example_id
        if mask.any():
            df.loc[mask, "human_esci_label"] = None
        message = "Cleared your label"
    else:
        # CSV mode: use index-based approach
        if index < 0 or index >= len(df):
            return jsonify({"error": "Invalid index"}), 400

        df.at[index, "esci_label"] = ""
        df.to_csv(csv_file_path, index=False)
        message = "Cleared label in CSV"

    return jsonify({"success": True, "message": message})


@app.route("/api/copy-ai-label", methods=["POST"])
def copy_ai_label():
    """Copy AI label to human label"""
    global df, use_database

    data = request.json
    example_id = data.get("example_id")

    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    # Only works in database mode
    if not use_database:
        return jsonify({"error": "Copy AI label only works in database mode"}), 400

    # Database mode: require example_id for proper record identification
    if not example_id:
        return jsonify({"error": "example_id required for database mode"}), 400

    # Get the AI label
    mask = df["example_id"] == example_id
    if not mask.any():
        return jsonify({"error": "Example not found"}), 400

    ai_label = df.loc[mask, "ai_esci_label"].iloc[0]
    if not ai_label:
        return jsonify({"error": "No AI label to copy"}), 400

    # Set the human label to the AI label value
    if not save_label_to_database_by_id(example_id, ai_label):
        return jsonify({"error": "Failed to save label to database"}), 500

    # Update DataFrame
    df.loc[mask, "human_esci_label"] = ai_label

    return jsonify({"success": True, "message": f"Copied AI label: {ai_label}"})


def load_csv(file_path):
    """Load CSV file into global DataFrame"""
    global df, csv_file_path

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
        csv_file_path = file_path
        print(f"Loaded {len(df)} records from {file_path}")
        return True
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)


def get_labeler_id_from_name(labeler_name):
    """Get labeler ID from labeler name in database"""
    query = "SELECT id FROM labeler WHERE name = %s"
    try:
        result = execute_query(query, params=(labeler_name,))
        if result.empty:
            print(f"Error: Labeler '{labeler_name}' not found in database")
            sys.exit(1)
        return int(result.iloc[0]["id"])
    except Exception as e:
        print(f"Error getting labeler ID: {e}")
        sys.exit(1)


def load_database_data(run_id, labeler_name):
    """Load data from database for specific MLflow run"""
    global df, mlflow_run_id, labeler_id

    try:
        # Get labeler ID for human labeler
        labeler_id = get_labeler_id_from_name(labeler_name)
        mlflow_run_id = run_id

        # Query to get examples with both AI and human labels
        query = """
        SELECT
            e.id as example_id,
            e.query_id,
            e.consumable_id,
            e.example_gen_hash,
            q.query_content as query,
            q.created_at,
            c.consumable_name,
            c.consumable_ingredients,
            ai_label.esci_label as ai_esci_label,
            human_label.esci_label as human_esci_label
        FROM example e
        JOIN query q ON e.query_id = q.id
        JOIN consumable c ON e.consumable_id = c.id
        LEFT JOIN label ai_label ON e.id = ai_label.example_id
            AND ai_label.labeler_id = (
                SELECT id FROM labeler WHERE name = e.example_gen_hash
            )
        LEFT JOIN label human_label ON e.id = human_label.example_id
            AND human_label.labeler_id = %s
        WHERE q.mlflow_run_id = %s
        ORDER BY e.id
        """

        df = execute_query(query, params=(labeler_id, run_id))

        if df.empty:
            print(f"Error: No examples found for MLflow run '{run_id}'")
            sys.exit(1)

        print(f"Loaded {len(df)} examples from database for run {run_id}")

        # Detect if this is an E-focused run and apply filtering
        ai_label_counts = df["ai_esci_label"].value_counts()
        total_ai_labels = ai_label_counts.sum()

        if total_ai_labels > 0:
            e_percentage = (ai_label_counts.get("E", 0) / total_ai_labels) * 100
            if e_percentage > 70:  # 70% threshold for E-focused runs
                print(f"🟢 E-focused run detected ({e_percentage:.1f}% E labels)")

                # Filter out trivial matches where query == consumable_name
                before_count = len(df)
                df = df[df["query"].str.lower() != df["consumable_name"].str.lower()]
                df = df.reset_index(drop=True)  # Reset indices after filtering
                filtered_count = before_count - len(df)

                if filtered_count > 0:
                    print(
                        f"🚫 Filtered out {filtered_count} trivial matches "
                        f"(query == consumable_name)"
                    )

                if df.empty:
                    print("Error: No examples remaining after filtering")
                    sys.exit(1)

        print(f"Labeler: {labeler_name} (ID: {labeler_id})")

        # Show label statistics
        ai_labeled = df["ai_esci_label"].notna().sum()
        human_labeled = df["human_esci_label"].notna().sum()
        print(f"AI labels: {ai_labeled}, Human labels: {human_labeled}")

        return True

    except Exception as e:
        print(f"Error loading database data: {e}")
        sys.exit(1)


def save_label_to_database(example_index, esci_label):
    """Save ESCI label to database"""
    global df, labeler_id

    try:
        example_id = int(df.iloc[example_index]["example_id"])

        # Insert or update label in database
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # First try to update existing label
                cursor.execute(
                    """
                    UPDATE label
                    SET esci_label = %s, created_at = NOW()
                    WHERE example_id = %s AND labeler_id = %s
                """,
                    (esci_label, example_id, labeler_id),
                )

                # If no rows updated, insert new label
                if cursor.rowcount == 0:
                    cursor.execute(
                        """
                        INSERT INTO label (labeler_id, example_id, esci_label)
                        VALUES (%s, %s, %s)
                    """,
                        (labeler_id, example_id, esci_label),
                    )

                conn.commit()

        return True

    except Exception as e:
        print(f"Database save error: {e}")
        return False


def save_label_to_database_by_id(example_id, esci_label):
    """Save ESCI label to database by example_id"""
    global labeler_id

    try:
        # Insert or update label in database
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # First try to update existing label
                cursor.execute(
                    """
                    UPDATE label
                    SET esci_label = %s, created_at = NOW()
                    WHERE example_id = %s AND labeler_id = %s
                """,
                    (esci_label, example_id, labeler_id),
                )

                # If no rows updated, insert new label
                if cursor.rowcount == 0:
                    cursor.execute(
                        """
                        INSERT INTO label (labeler_id, example_id, esci_label)
                        VALUES (%s, %s, %s)
                    """,
                        (labeler_id, example_id, esci_label),
                    )

                conn.commit()

        return True

    except Exception as e:
        print(f"Database save error: {e}")
        return False


def update_query_in_database(example_index, new_query_text):
    """Update query text with global deduplication"""
    global df

    try:
        example_id = int(df.iloc[example_index]["example_id"])
        current_query_id = int(df.iloc[example_index]["query_id"])

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Check if the new query text already exists globally
                cursor.execute(
                    "SELECT id FROM query WHERE query_content = %s LIMIT 1",
                    (new_query_text,),
                )
                existing_query = cursor.fetchone()

                if existing_query:
                    # Reuse existing query - update example to point to it
                    existing_query_id = existing_query[0]
                    cursor.execute(
                        """
                        UPDATE example SET query_id = %s WHERE id = %s
                    """,
                        (existing_query_id, example_id),
                    )

                    # Update our DataFrame
                    df.at[example_index, "query_id"] = existing_query_id
                    df.at[example_index, "query"] = new_query_text

                else:
                    # Update current query text (first occurrence owns metadata)
                    cursor.execute(
                        """
                        UPDATE query SET query_content = %s WHERE id = %s
                    """,
                        (new_query_text, current_query_id),
                    )

                    # Update our DataFrame
                    df.at[example_index, "query"] = new_query_text

                conn.commit()

        return True

    except Exception as e:
        print(f"Database update error: {e}")
        return False


def update_query_in_database_by_id(example_id, new_query_text):
    """Update query text with global deduplication by example_id"""
    global df

    try:
        # Find the current query_id for this example
        mask = df["example_id"] == example_id
        if not mask.any():
            return False

        current_query_id = int(df.loc[mask, "query_id"].iloc[0])

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Check if the new query text already exists globally
                cursor.execute(
                    "SELECT id FROM query WHERE query_content = %s LIMIT 1",
                    (new_query_text,),
                )
                existing_query = cursor.fetchone()

                if existing_query:
                    # Reuse existing query - update example to point to it
                    existing_query_id = existing_query[0]
                    cursor.execute(
                        """
                        UPDATE example SET query_id = %s WHERE id = %s
                    """,
                        (existing_query_id, example_id),
                    )

                    # Update our DataFrame
                    df.loc[mask, "query_id"] = existing_query_id
                    df.loc[mask, "query"] = new_query_text

                else:
                    # Update current query text (first occurrence owns metadata)
                    cursor.execute(
                        """
                        UPDATE query SET query_content = %s WHERE id = %s
                    """,
                        (new_query_text, current_query_id),
                    )

                    # Update our DataFrame
                    df.loc[mask, "query"] = new_query_text

                conn.commit()

        return True

    except Exception as e:
        print(f"Database update error: {e}")
        return False


def delete_example_from_database(example_index):
    """Delete example and all its labels"""
    global df

    try:
        example_id = int(df.iloc[example_index]["example_id"])
        query_id = int(df.iloc[example_index]["query_id"])

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Delete labels first (foreign key constraint)
                cursor.execute("DELETE FROM label WHERE example_id = %s", (example_id,))
                labels_deleted = cursor.rowcount

                # Delete the example
                cursor.execute("DELETE FROM example WHERE id = %s", (example_id,))

                # Check if this query has any remaining examples
                cursor.execute(
                    "SELECT COUNT(*) FROM example WHERE query_id = %s", (query_id,)
                )
                remaining_examples = cursor.fetchone()[0]

                # If no examples left, delete the orphaned query
                if remaining_examples == 0:
                    cursor.execute("DELETE FROM query WHERE id = %s", (query_id,))
                    print(f"Deleted orphaned query {query_id}")

                conn.commit()

        # Remove from DataFrame
        df.drop(df.index[example_index], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return True, labels_deleted

    except Exception as e:
        print(f"Database delete error: {e}")
        return False, 0


def delete_example_from_database_by_id(example_id):
    """Delete example and all its labels by example_id"""
    global df

    try:
        # Find the query_id for this example
        mask = df["example_id"] == example_id
        if not mask.any():
            return False, 0

        query_id = int(df.loc[mask, "query_id"].iloc[0])

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Delete labels first (foreign key constraint)
                cursor.execute("DELETE FROM label WHERE example_id = %s", (example_id,))
                labels_deleted = cursor.rowcount

                # Delete the example
                cursor.execute("DELETE FROM example WHERE id = %s", (example_id,))

                # Check if this query has any remaining examples
                cursor.execute(
                    "SELECT COUNT(*) FROM example WHERE query_id = %s", (query_id,)
                )
                remaining_examples = cursor.fetchone()[0]

                # If no examples left, delete the orphaned query
                if remaining_examples == 0:
                    cursor.execute("DELETE FROM query WHERE id = %s", (query_id,))
                    print(f"Deleted orphaned query {query_id}")

                conn.commit()

        # Remove from DataFrame
        df.drop(df.index[mask], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return True, labels_deleted

    except Exception as e:
        print(f"Database delete error: {e}")
        return False, 0


def delete_label_from_database(example_index):
    """Delete only the current labeler's label for this example"""
    global df, labeler_id

    try:
        example_id = int(df.iloc[example_index]["example_id"])

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Delete only this labeler's label
                cursor.execute(
                    """
                    DELETE FROM label
                    WHERE example_id = %s AND labeler_id = %s
                """,
                    (example_id, labeler_id),
                )

                conn.commit()

        # Update DataFrame - clear human label
        df.at[example_index, "human_esci_label"] = None
        # Recalculate combined label (fall back to AI label)
        ai_label = df.iloc[example_index]["ai_esci_label"]
        df.at[example_index, "esci_label"] = ai_label or ""

        return True

    except Exception as e:
        print(f"Database delete error: {e}")
        return False


def delete_label_from_database_by_id(example_id):
    """Delete only the current labeler's label for this example by example_id"""
    global labeler_id

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Delete only this labeler's label
                cursor.execute(
                    """
                    DELETE FROM label
                    WHERE example_id = %s AND labeler_id = %s
                """,
                    (example_id, labeler_id),
                )

                conn.commit()

        return True

    except Exception as e:
        print(f"Database delete error: {e}")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ESCI Annotation Tool - Annotate query-consumable pairs"
    )

    # Mutually exclusive group for data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("csv_file", nargs="?", help="Path to CSV file to annotate")
    data_group.add_argument(
        "--database", action="store_true", help="Use database mode instead of CSV file"
    )

    # Database-specific arguments
    parser.add_argument(
        "--run-id",
        help="MLflow run ID to load examples from (required with --database)",
    )
    parser.add_argument(
        "--labeler-name", default="Luv", help="Name of the labeler (default: Luv)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.database:
        if not args.run_id:
            print("Error: --run-id is required when using --database")
            sys.exit(1)

        use_database = True
        load_database_data(args.run_id, args.labeler_name)

        print("🏷️ ESCI Annotation Tool starting...")
        print("🗄️ Mode: Database")
        print(f"📊 MLflow Run: {args.run_id}")
        print(f"👤 Labeler: {args.labeler_name}")
        print(f"📊 Records: {len(df)}")

    else:
        if not args.csv_file:
            print("Error: CSV file path is required when not using --database")
            sys.exit(1)

        use_database = False
        load_csv(args.csv_file)

        print("🏷️ ESCI Annotation Tool starting...")
        print("📁 Mode: CSV File")
        print(f"📁 File: {args.csv_file}")
        print(f"📊 Records: {len(df)}")

    print("🌐 Open: http://localhost:5002")
    print("⌨️  Keyboard: Z=E, X=S, C=C, V=I")

    app.run(debug=True, port=5002)
