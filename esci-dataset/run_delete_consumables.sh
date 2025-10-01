#!/bin/bash

# Delete Consumables Script Runner

# Edit the parameters below, then run: ./run_delete_consumables.sh

# Choose mode: "IDS", "PATTERN", or "PATTERN_LIST"
MODE="IDS"

# IDS mode parameters (used when MODE="IDS")
# Space-separated list of consumable IDs
CONSUMABLE_IDS=$(cat consumable_ids.txt)

# PATTERN mode parameters (used when MODE="PATTERN")
# Pattern to match in consumable names (case-insensitive)
PATTERN="mixed platter"
# Set to "true" for exact match, "false" for substring match
EXACT_MATCH="true"

# PATTERN_LIST mode parameters (used when MODE="PATTERN_LIST")
# Comma-separated list of patterns to match
PATTERN_LIST="mixed,assorted,platter,combo"

# Action: "DRY_RUN" or "CONFIRM"
# Use DRY_RUN to preview changes without making them
ACTION="DRY_RUN"

# Display info
echo "🗑️  Delete Consumables Tool"
echo ""

if [ "$MODE" = "IDS" ]; then
    echo "🔢 Mode: Delete by IDs"
    echo "📋 IDs: $CONSUMABLE_IDS"
    echo ""

    if [ "$ACTION" = "DRY_RUN" ]; then
        echo "👁️  DRY RUN - Preview only (no changes will be made)"
        PYTHONPATH=. uv run python database/scripts/delete_consumables.py --ids $CONSUMABLE_IDS --dry-run
    elif [ "$ACTION" = "CONFIRM" ]; then
        echo "⚠️  CONFIRM - This will DELETE data!"
        PYTHONPATH=. uv run python database/scripts/delete_consumables.py --ids $CONSUMABLE_IDS --confirm
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
        PYTHONPATH=. uv run python database/scripts/delete_consumables.py --pattern "$PATTERN" $EXACT_FLAG --dry-run
    elif [ "$ACTION" = "CONFIRM" ]; then
        echo "⚠️  CONFIRM - This will DELETE data!"
        PYTHONPATH=. uv run python database/scripts/delete_consumables.py --pattern "$PATTERN" $EXACT_FLAG --confirm
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
        PYTHONPATH=. uv run python database/scripts/delete_consumables.py --pattern-list "$PATTERN_LIST" --dry-run
    elif [ "$ACTION" = "CONFIRM" ]; then
        echo "⚠️  CONFIRM - This will DELETE data!"
        PYTHONPATH=. uv run python database/scripts/delete_consumables.py --pattern-list "$PATTERN_LIST" --confirm
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