#!/bin/bash

# Synthetic Data Rollback Runner
# Edit the defaults below, then run: ./run_rollback_synthetic_data.sh

# Default parameters (edit these as needed)
RUN_ID=""
DRY_RUN="true"
CONFIRM=""

uv run python database/scripts/rollback_synthetic_data.py \
	$(if [ -n "$RUN_ID" ]; then echo "--run-id $RUN_ID"; fi) \
	$(if [ "$DRY_RUN" = "true" ]; then echo "--dry-run"; fi) \
	$(if [ "$CONFIRM" = "true" ]; then echo "--confirm"; fi)