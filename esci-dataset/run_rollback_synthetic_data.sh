#!/bin/bash

# Synthetic Data Rollback Runner
# Edit the defaults below, then run: ./run_rollback_synthetic_data.sh

# Default parameters (edit these as needed)
RUN_ID="aefc21fcce0440a28943f874c04dd37d"
DRY_RUN=""
CONFIRM="true"

uv run python database/scripts/rollback_synthetic_data.py \
	$(if [ -n "$RUN_ID" ]; then echo "--run-id $RUN_ID"; fi) \
	$(if [ "$DRY_RUN" = "true" ]; then echo "--dry-run"; fi) \
	$(if [ "$CONFIRM" = "true" ]; then echo "--confirm"; fi)