#!/bin/bash

# Synthetic Data Migration Script Runner
# Edit the defaults below, then run: ./run_migrate_synthetic_data.sh

# Default parameters (edit these as needed)
EXPERIMENT_NAME=""
RUN_ID=""
ALL_APPROVED=1
DRY_RUN=""
CONFIRM="1"

uv run python database/scripts/migrate_synthetic_data.py \
	$(if [ -n "$EXPERIMENT_NAME" ]; then echo "--experiment-name $EXPERIMENT_NAME"; fi) \
	$(if [ -n "$RUN_ID" ]; then echo "--run-id $RUN_ID"; fi) \
	$(if [ -n "$ALL_APPROVED" ]; then echo "--all-approved"; fi) \
	$(if [ -n "$DRY_RUN" ]; then echo "--dry-run"; fi) \
	$(if [ -n "$CONFIRM" ]; then echo "--confirm"; fi)