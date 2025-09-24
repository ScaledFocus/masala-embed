#!/bin/bash

# MLflow Runs Summary Script Runner
# Edit the defaults below, then run: ./run_summary.sh

# Default parameters (edit these as needed)
EXPERIMENT_NAME=""  # Leave empty for all experiments, or set to specific experiment like "Intent_Generation"
OUTPUT="mlflow_migrated_runs_summary.csv"  # Output filename
OUTPUT_DIR="output/summary"  # Output directory

uv run python scripts/mlflow_runs_summary.py \
	$(if [ -n "$EXPERIMENT_NAME" ]; then echo "--experiment-name $EXPERIMENT_NAME"; fi) \
	--output "$OUTPUT" \
	--output-dir "$OUTPUT_DIR"