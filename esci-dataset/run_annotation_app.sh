#!/bin/bash

# ESCI Annotation App Runner
# Usage: ./run_annotation_app.sh <csv_file>

set -e

# Check if CSV file argument provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide a CSV file"
    echo "Usage: $0 <csv_file>"
    echo ""
    echo "Available CSV files:"
    find output/ -name "*.csv" 2>/dev/null | head -5
    exit 1
fi

CSV_FILE=$1

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ Error: CSV file '$CSV_FILE' not found"
    exit 1
fi

# Display info
echo "🏷️  Starting ESCI Annotation Tool"
echo "📁 File: $CSV_FILE"
echo "🌐 URL: http://localhost:5000"
echo "⌨️  Shortcuts: Z=E, X=S, C=C, V=I, ←→=Navigate"
echo "📍 Skip to record: Type number + Enter/Go"
echo ""

# Run the Flask app
uv run python annotation/app.py "$CSV_FILE"