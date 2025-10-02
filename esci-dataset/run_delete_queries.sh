#!/bin/bash

# Delete Queries Script Runner

# Edit the parameters below, then run: ./run_delete_queries.sh

# Choose mode: "IDS", "PATTERN", or "PATTERN_LIST"
MODE="PATTERN"

# IDS mode parameters (used when MODE="IDS")
# Space-separated list of query IDs
QUERY_IDS="1001 1002 1003"

# PATTERN mode parameters (used when MODE="PATTERN")
# Pattern to match in query content (case-insensitive)
PATTERN="pizza"
# Set to "true" for exact match, "false" for substring match
EXACT_MATCH="false"

# PATTERN_LIST mode parameters (used when MODE="PATTERN_LIST")
# Comma-separated list of patterns to match
PATTERN_LIST="food,meal,dish,something"

# Action: "DRY_RUN" or "CONFIRM"
# Use DRY_RUN to preview changes without making them
ACTION="DRY_RUN"

# Display info
echo "🗑️  Delete Queries Tool"
echo ""

if [ "$MODE" = "IDS" ]; then
    echo "🔢 Mode: Delete by IDs"
    echo "📋 IDs: $QUERY_IDS"
    echo ""

    if [ "$ACTION" = "DRY_RUN" ]; then
        echo "👁️  DRY RUN - Preview only (no changes will be made)"
        PYTHONPATH=. uv run python database/scripts/delete_queries.py --ids $QUERY_IDS --dry-run
    elif [ "$ACTION" = "CONFIRM" ]; then
        echo "⚠️  CONFIRM - This will DELETE data!"
        PYTHONPATH=. uv run python database/scripts/delete_queries.py --ids $QUERY_IDS --confirm
    else
        echo "❌ Error: ACTION must be 'DRY_RUN' or 'CONFIRM'"
        exit 1
    fi

elif [ "$MODE" = "PATTERN" ]; then
    echo "🔍 Mode: Delete by Pattern"
    echo "📝 Pattern: '$PATTERN'"

    if [ "$EXACT_MATCH" = "true" ]; then
        echo "✅ Match Type: Exact"
        EXACT_FLAG="--exact"
    else
        echo "🔤 Match Type: Substring (contains)"
        EXACT_FLAG=""
    fi
    echo ""

    if [ "$ACTION" = "DRY_RUN" ]; then
        echo "👁️  DRY RUN - Preview only (no changes will be made)"
        PYTHONPATH=. uv run python database/scripts/delete_queries.py --pattern "$PATTERN" $EXACT_FLAG --dry-run
    elif [ "$ACTION" = "CONFIRM" ]; then
        echo "⚠️  CONFIRM - This will DELETE data!"
        PYTHONPATH=. uv run python database/scripts/delete_queries.py --pattern "$PATTERN" $EXACT_FLAG --confirm
    else
        echo "❌ Error: ACTION must be 'DRY_RUN' or 'CONFIRM'"
        exit 1
    fi

elif [ "$MODE" = "PATTERN_LIST" ]; then
    echo "📋 Mode: Delete by Pattern List"
    echo "🔤 Patterns: $PATTERN_LIST"
    echo "🔤 Match Type: Substring (contains any pattern)"
    echo ""

    if [ "$ACTION" = "DRY_RUN" ]; then
        echo "👁️  DRY RUN - Preview only (no changes will be made)"
        PYTHONPATH=. uv run python database/scripts/delete_queries.py --pattern-list "$PATTERN_LIST" --dry-run
    elif [ "$ACTION" = "CONFIRM" ]; then
        echo "⚠️  CONFIRM - This will DELETE data!"
        PYTHONPATH=. uv run python database/scripts/delete_queries.py --pattern-list "$PATTERN_LIST" --confirm
    else
        echo "❌ Error: ACTION must be 'DRY_RUN' or 'CONFIRM'"
        exit 1
    fi

else
    echo "❌ Error: MODE must be 'IDS', 'PATTERN', or 'PATTERN_LIST'"
    echo "Edit the script and set MODE to one of these values"
    exit 1
fi

echo ""
echo "✅ Done!"
