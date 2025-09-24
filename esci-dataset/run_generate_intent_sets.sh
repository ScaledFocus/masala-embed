#!/bin/bash

# Intent Set Generator Runner
# Edit the defaults below, then run: ./run_generate_intent_sets.sh

# Default parameters (edit these as needed)
OUTPUT_DIR="intent_sets"
NUM_SETS=10
INTENTS_PER_THEME=100
MODEL="gpt-5"
TEMPERATURE=1.0
BASE_PROMPT=""

# Make script executable and run
chmod +x run_generate_intent_sets.sh

echo "🎯 Generating Intent Sets..."
echo "📁 Output Directory: $OUTPUT_DIR"
echo "🔢 Number of Sets: $NUM_SETS"
echo "📊 Intents per Theme: $INTENTS_PER_THEME (x10 themes = $((INTENTS_PER_THEME * 10)) total)"
echo "🤖 Model: $MODEL"
echo ""

uv run python src/data_generation/generate_intent_sets.py \
	--output-dir $OUTPUT_DIR \
	--num-sets $NUM_SETS \
	--intents-per-theme $INTENTS_PER_THEME \
	--model $MODEL \
	--temperature $TEMPERATURE \
	$(if [ -n "$BASE_PROMPT" ]; then echo "--base-prompt $BASE_PROMPT"; fi)