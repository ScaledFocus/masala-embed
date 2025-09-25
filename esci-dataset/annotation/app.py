#!/usr/bin/env python3
"""
Flask-based ESCI Annotation Tool
Directly modifies the CSV file - no downloads needed!

Usage:
    python app.py queries_I_batch100_limit200_v2_20250925_112245.csv
"""

import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Global variables
csv_file_path = None
df = None


@app.route("/")
def index():
    return render_template("annotate.html")


@app.route("/api/data")
def get_data():
    """Get all annotation data"""
    global df
    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    return jsonify(
        {
            "data": df.to_dict("records"),
            "total": len(df),
            "filename": os.path.basename(csv_file_path),
        }
    )


@app.route("/api/update", methods=["POST"])
def update_label():
    """Update ESCI label for a specific row"""
    global df, csv_file_path

    data = request.json
    index = data.get("index")
    label = data.get("label")

    if df is None:
        return jsonify({"error": "No data loaded"}), 400

    if index < 0 or index >= len(df):
        return jsonify({"error": "Invalid index"}), 400

    if label not in ["E", "S", "C", "I"]:
        return jsonify({"error": "Invalid label"}), 400

    # Update the DataFrame
    df.at[index, "esci_label"] = label

    # Save to CSV immediately
    df.to_csv(csv_file_path, index=False)

    return jsonify(
        {"success": True, "message": f"Updated row {index} to label {label}"}
    )


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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <csv_file_path>")
        sys.exit(1)

    csv_file = sys.argv[1]
    load_csv(csv_file)

    print("🏷️ ESCI Annotation Tool starting...")
    print(f"📁 File: {csv_file}")
    print(f"📊 Records: {len(df)}")
    print("🌐 Open: http://localhost:5000")
    print("⌨️  Keyboard: Z=E, X=S, C=C, V=I")

    app.run(debug=True, port=5000)
