#!/bin/bash

# Synthetic Data Migration Script Runner
# Edit the defaults below, then run: ./run_migrate_synthetic_data.sh

# Default parameters (edit these as needed)
EXPERIMENT_NAME="Intent_Generation"
RUN_ID="bc54925269c74aaba03741b1214c7766"
ALL_APPROVED=""
DRY_RUN=""
CONFIRM="true"
BRIDGED="true"

uv run python database/scripts/migrate_synthetic_data.py \
	$(if [ -n "$EXPERIMENT_NAME" ]; then echo "--experiment-name $EXPERIMENT_NAME"; fi) \
	$(if [ -n "$RUN_ID" ]; then echo "--run-id $RUN_ID"; fi) \
	$(if [ -n "$ALL_APPROVED" ]; then echo "--all-approved"; fi) \
	$(if [ -n "$DRY_RUN" ]; then echo "--dry-run"; fi) \
	$(if [ -n "$CONFIRM" ]; then echo "--confirm"; fi) \
	$(if [ -n "$BRIDGED" ]; then echo "--bridged"; fi)