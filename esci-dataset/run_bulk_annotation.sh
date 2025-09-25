#!/bin/bash

# ESCI Bulk Annotation Tool Runner
# Usage:
#   ./run_bulk_annotation.sh <csv_file>
#   ./run_bulk_annotation.sh --database --run-id <run_id> [--labeler-id <name>]

set -e

# Check arguments
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide arguments"
    echo ""
    echo "Usage:"
    echo "  $0 <csv_file>                           # CSV mode"
    echo "  $0 --database --run-id <run_id>        # Database mode"
    echo "  $0 --database --run-id <run_id> --labeler-id <name>"
    echo ""
    echo "Available CSV files:"
    find output/ -name "*.csv" 2>/dev/null | head -5
    exit 1
fi

# Display info
echo "🚀 Starting ESCI Bulk Annotation Tool"

# Check if database mode
if [ "$1" = "--database" ]; then
    echo "🗄️ Mode: Database"
    echo "🌐 URL: http://localhost:5003"
else
    CSV_FILE=$1
    # Check if CSV file exists
    if [ ! -f "$CSV_FILE" ]; then
        echo "❌ Error: CSV file '$CSV_FILE' not found"
        exit 1
    fi
    echo "📁 Mode: CSV File"
    echo "📁 File: $CSV_FILE"
    echo "🌐 URL: http://localhost:5003"
fi

echo "⚡ Fast Mode: Multi-record view with adjustable page size"
echo "⌨️  Shortcuts: Click E/S/C/I buttons on each card"
echo ""

# Run the Flask app with all arguments
uv run python annotation/app_bulk.py "$@"