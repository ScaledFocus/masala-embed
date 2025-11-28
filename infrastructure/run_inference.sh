#!/bin/bash

set -e

set -a
source .env
set +a

echo "Checking if required environment variables are set."

# Check for required parameters
if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME not set (e.g., 'qwen', 'siglip', 'colqwen')"
    exit 1
fi

if [ -z "$MODEL_TYPE" ]; then
    echo "Error: MODEL_TYPE not set (e.g., 'text_embedding', 'multimodal')"
    exit 1
fi

if [ -z "$MODEL_SOURCE" ]; then
    echo "Error: MODEL_SOURCE not set (e.g., 'r2', 'huggingface')"
    exit 1
fi

# Check source-specific requirements
if [ "$MODEL_SOURCE" = "r2" ]; then
    if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
        echo "Error: CLOUDFLARE_ACCOUNT_ID not set (required for R2 models)"
        exit 1
    fi

    if [ -z "$R2_ACCESS_KEY_ID" ]; then
        echo "Error: R2_ACCESS_KEY_ID not set (required for R2 models)"
        exit 1
    fi

    if [ -z "$R2_SECRET_ACCESS_KEY" ]; then
        echo "Error: R2_SECRET_ACCESS_KEY not set (required for R2 models)"
        exit 1
    fi

    if [ -z "$R2_MODEL_FILE" ]; then
        echo "Error: R2_MODEL_FILE not set (required for R2 models)"
        exit 1
    fi
elif [ "$MODEL_SOURCE" = "huggingface" ]; then
    if [ -z "$HF_MODEL_ID" ]; then
        echo "Error: HF_MODEL_ID not set (required for HuggingFace models)"
        exit 1
    fi
fi

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

echo "Starting model inference deployment..."
echo "Model Name: $MODEL_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Model Source: $MODEL_SOURCE"

if [ "$MODEL_SOURCE" = "r2" ]; then
    echo "R2 File: $R2_MODEL_FILE"
elif [ "$MODEL_SOURCE" = "huggingface" ]; then
    echo "HF Model ID: $HF_MODEL_ID"
fi

# Deploy the model
modal deploy inference.py

echo "Model deployment completed successfully!"
