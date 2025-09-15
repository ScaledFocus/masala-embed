#!/bin/bash

set -e

set -a
source .env
set +a

echo "Checking if required environment variables are set."

# Check for Cloudflare R2 credentials
if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
    echo "Error: CLOUDFLARE_ACCOUNT_ID not set"
    exit 1
fi

if [ -z "$R2_ACCESS_KEY_ID" ]; then
    echo "Error: R2_ACCESS_KEY_ID not set"
    exit 1
fi

if [ -z "$R2_SECRET_ACCESS_KEY" ]; then
    echo "Error: R2_SECRET_ACCESS_KEY not set"
    exit 1
fi

# Check for required parameters
if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME not set (e.g., 'Qwen/Qwen2.5-0.5B')"
    exit 1
fi

if [ -z "$QUANTIZATION_TYPE" ]; then
    echo "Error: QUANTIZATION_TYPE not set (e.g., 'Q4_0', 'Q8_0')"
    exit 1
fi

# Optional parameters with defaults
PRECISION=${PRECISION:-"f16"}
R2_BUCKET_NAME=${R2_BUCKET_NAME:-"masala-embed-models"}

# Check if modal is installed
if ! command -v modal >/dev/null 2>&1; then
    echo "Error: modal is not installed or not in PATH"
    echo "Install with: pip install modal"
    exit 1
fi

# Check if modal is authenticated
if ! modal token set --help >/dev/null 2>&1; then
    echo "Error: modal CLI is not properly configured"
    echo "Run: modal setup"
    exit 1
fi

cd scripts

echo "Starting model quantization..."
echo "Model: $MODEL_NAME"
echo "Quantization: $QUANTIZATION_TYPE"
echo "Precision: $PRECISION"
echo "Bucket: $R2_BUCKET_NAME"

# Run the quantization
modal run quantize.py \
    --model-name "$MODEL_NAME" \
    --quantization-type "$QUANTIZATION_TYPE" \
    --precision "$PRECISION" \

echo "Quantization completed successfully!"
