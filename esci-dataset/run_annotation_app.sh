#!/bin/bash

# ESCI Annotation App Runner

# Edit the parameters below, then run: ./run_annotation_app.sh

# Choose mode: "CSV" or "DATABASE"
MODE="DATABASE"

# CSV mode parameters (used when MODE="CSV")
CSV_FILE=""

# Database mode parameters (used when MODE="DATABASE")
RUN_ID="52a7259b2e854d7e9838568133a38d63"
LABELER_NAME="Luv"

# Display info
echo "🏷️  Starting ESCI Annotation Tool"

if [ "$MODE" = "DATABASE" ]; then
    echo "🗄️ Mode: Database"
    echo "📊 Run ID: $RUN_ID"
    echo "👤 Labeler: $LABELER_NAME"
    echo "🌐 URL: http://localhost:5002"
    echo "⌨️  Shortcuts: Z=E, X=S, C=C, V=I, ←→=Navigate, Esc=Clear, Enter=Copy AI"
    echo ""

    uv run python annotation/app.py --database --run-id "$RUN_ID" --labeler-name "$LABELER_NAME"

elif [ "$MODE" = "CSV" ]; then
    echo "📁 Mode: CSV File"
    echo "📁 File: $CSV_FILE"
    echo "🌐 URL: http://localhost:5002"
    echo "⌨️  Shortcuts: Z=E, X=S, C=C, V=I, ←→=Navigate"
    echo ""

    # Check if CSV file exists
    if [ ! -f "$CSV_FILE" ]; then
        echo "❌ Error: CSV file '$CSV_FILE' not found"
        echo "Available CSV files:"
        find output/ -name "*.csv" 2>/dev/null | head -5
        exit 1
    fi

    uv run python annotation/app.py "$CSV_FILE"

else
    echo "❌ Error: MODE must be 'CSV' or 'DATABASE'"
    echo "Edit the script and set MODE=\"CSV\" or MODE=\"DATABASE\""
    exit 1
fi