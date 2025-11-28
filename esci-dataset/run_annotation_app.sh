#!/bin/bash

# ESCI Annotation App Runner

# Edit the parameters below, then run: ./run_annotation_app.sh

# Choose mode: "CSV" or "DATABASE"
MODE="CSV"

# CSV mode parameters (used when MODE="CSV")
CSV_FILE="/Users/luvsuneja/Documents/repos/masala-embed/esci-dataset/mlruns/180351575577738277/c3e73bf4cd934d1d981e57774cfd2d20/artifacts/outputs/queries_E_batch50_limit500_start3000_v3_20251002_160353.csv"

# Database mode parameters (used when MODE="DATABASE")
RUN_ID="0c611a39b1844107b10d4f7ac9282770"
LABELER_NAME="Luv"

# Display info
echo "🏷️  Starting ESCI Annotation Tool"

if [ "$MODE" = "DATABASE" ]; then
    echo "🗄️ Mode: Database"
    echo "📊 Run ID: $RUN_ID"
    echo "👤 Labeler: $LABELER_NAME"
    echo "🌐 URL: http://localhost:5002"
    echo "⌨️  Shortcuts: Z=E, X=S, C=C, V=I, ←→=Navigate, Esc=Clear, Enter=Copy AI, R=Review Mode"
    echo ""

    uv run python annotation/app.py --database --run-id "$RUN_ID" --labeler-name "$LABELER_NAME"

elif [ "$MODE" = "CSV" ]; then
    echo "📁 Mode: CSV File"
    echo "📁 File: $CSV_FILE"
    echo "🌐 URL: http://localhost:5002"
    echo "⌨️  Shortcuts: Z=E, X=S, C=C, V=I, ←→=Navigate, R=Review Mode"
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