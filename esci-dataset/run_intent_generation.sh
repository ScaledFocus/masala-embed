#!/bin/bash

# Intent Generation MLflow Wrapper Runner
# Edit the defaults below, then run: ./run_intent_generation.sh

# Default parameters (edit these as needed)
MODEL="gpt-5-mini"
NUM_INTENTS=50
LIMIT=20
BATCH_SIZE=10
QUERIES_PER_ITEM=2
STOP_AT_INTENTS=""
DIETARY_FLAG=""
TEMPERATURE=1.0
OUTPUT_DIR="output"
EXPERIMENT_NAME="initial-generation-test"
RUN_NAME=""
STEP1_PROMPT=""
STEP2_PROMPT=""
STEP3_PROMPT=""

uv run python src/mlflow_wrapper/mlflow_intent_generation.py \
	--model $MODEL \
	--num-intents $NUM_INTENTS \
	--batch-size $BATCH_SIZE \
	--queries-per-item $QUERIES_PER_ITEM \
	--temperature $TEMPERATURE \
	--output-dir $OUTPUT_DIR \
	$(if [ -n "$LIMIT" ]; then echo "--limit $LIMIT"; fi) \
	$(if [ -n "$STOP_AT_INTENTS" ]; then echo "--stop-at-intents"; fi) \
	$(if [ -n "$DIETARY_FLAG" ]; then echo "--dietary_flag"; fi) \
	$(if [ -n "$EXPERIMENT_NAME" ]; then echo "--experiment-name $EXPERIMENT_NAME"; fi) \
	$(if [ -n "$RUN_NAME" ]; then echo "--run-name $RUN_NAME"; fi) \
	$(if [ -n "$STEP1_PROMPT" ]; then echo "--step1-prompt $STEP1_PROMPT"; fi) \
	$(if [ -n "$STEP2_PROMPT" ]; then echo "--step2-prompt $STEP2_PROMPT"; fi) \
	$(if [ -n "$STEP3_PROMPT" ]; then echo "--step3-prompt $STEP3_PROMPT"; fi)