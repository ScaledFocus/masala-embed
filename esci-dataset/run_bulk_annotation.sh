#!/bin/bash

# ESCI Bulk Annotation Tool Runner

# Edit the parameters below, then run: ./run_bulk_annotation.sh

# Choose mode: "CSV" or "DATABASE"
MODE="DATABASE"

# CSV mode parameters (used when MODE="CSV")
CSV_FILE=""

# Database mode parameters (used when MODE="DATABASE")
RUN_ID="9f61abdd8bde40a28208f381504db898"
LABELER_NAME="Luv"

# UI parameters
RECORDS_PER_PAGE=20

# Display info
echo "🚀 Starting ESCI Bulk Annotation Tool"

if [ "$MODE" = "DATABASE" ]; then
    echo "🗄️ Mode: Database"
    echo "📊 Run ID: $RUN_ID"
    echo "👤 Labeler: $LABELER_NAME"
    echo "🌐 URL: http://localhost:5003"
    echo "📄 Page Size: $RECORDS_PER_PAGE records per page"
    echo "⚡ Fast Mode: Multi-record view with adjustable page size"
    echo "⌨️  Shortcuts: Click E/S/C/I buttons on each card"
    echo ""

    uv run python annotation/app_bulk.py --database --run-id "$RUN_ID" --labeler-name "$LABELER_NAME" --page-size "$RECORDS_PER_PAGE"

elif [ "$MODE" = "CSV" ]; then
    echo "📁 Mode: CSV File"
    echo "📁 File: $CSV_FILE"
    echo "🌐 URL: http://localhost:5003"
    echo "📄 Page Size: $RECORDS_PER_PAGE records per page"
    echo "⚡ Fast Mode: Multi-record view with adjustable page size"
    echo "⌨️  Shortcuts: Click E/S/C/I buttons on each card"
    echo ""

    # Check if CSV file exists
    if [ ! -f "$CSV_FILE" ]; then
        echo "❌ Error: CSV file '$CSV_FILE' not found"
        echo "Available CSV files:"
        find output/ -name "*.csv" 2>/dev/null | head -5
        exit 1
    fi

    uv run python annotation/app_bulk.py "$CSV_FILE" --page-size "$RECORDS_PER_PAGE"

else
    echo "❌ Error: MODE must be 'CSV' or 'DATABASE'"
    echo "Edit the script and set MODE=\"CSV\" or MODE=\"DATABASE\""
    exit 1
fi