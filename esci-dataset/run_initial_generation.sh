#!/bin/bash

# Initial Generation MLflow Wrapper Runner

# Edit the defaults below, then run: ./run_initial_generation.sh

# Default parameters (edit these as needed)
ESCI_LABEL="E"
MODEL="gpt-5"
BATCH_SIZE=50
QUERIES_PER_ITEM=2
TEMPERATURE=1.2
MAX_RETRIES=3
LIMIT=200
START_IDX=1000
PARALLEL=10
DIETARY_FLAG=""
OUTPUT_PATH=""
TEMPLATE_PATH=""
QUERY_EXAMPLES=""
EXPERIMENT_NAME="Initial_Generation"
RUN_NAME=""

uv run python src/mlflow_wrapper/mlflow_initial_generation.py \
	--esci_label $ESCI_LABEL \
	--model $MODEL \
	--batch_size $BATCH_SIZE \
	--queries_per_item $QUERIES_PER_ITEM \
	--temperature $TEMPERATURE \
	--max_retries $MAX_RETRIES \
	--parallel $PARALLEL \
	$(if [ -n "$LIMIT" ]; then echo "--limit $LIMIT"; fi) \
	$(if [ -n "$START_IDX" ]; then echo "--start_idx $START_IDX"; fi) \
	$(if [ -n "$DIETARY_FLAG" ]; then echo "--dietary_flag"; fi) \
	$(if [ -n "$OUTPUT_PATH" ]; then echo "--output_path $OUTPUT_PATH"; fi) \
	$(if [ -n "$TEMPLATE_PATH" ]; then echo "--template_path $TEMPLATE_PATH"; fi) \
	$(if [ -n "$QUERY_EXAMPLES" ]; then echo "--query_examples $QUERY_EXAMPLES"; fi) \
	$(if [ -n "$EXPERIMENT_NAME" ]; then echo "--experiment-name $EXPERIMENT_NAME"; fi) \
	$(if [ -n "$RUN_NAME" ]; then echo "--run-name $RUN_NAME"; fi)
