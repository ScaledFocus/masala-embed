#!/bin/bash

set -e

echo "Checking if required environment variables are set."
if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
    echo "Error: CLOUDFLARE_ACCOUNT_ID not set"
    exit 1
fi

if ! command -v wrangler >/dev/null 2>&1; then
    echo "Error: wrangler is not installed or not in PATH"
    exit 1
fi

if ! wrangler whoami >/dev/null 2>&1; then
    echo "Error: Wrangler is not logged in"
    exit 1
fi

cd terraform

echo "Initializing Terraform"
terraform init

echo "Planning Terraform deployment"
terraform plan \
    -var="cloudflare_api_token=$CLOUDFLARE_API_TOKEN" \
    -var="cloudflare_account_id=$CLOUDFLARE_ACCOUNT_ID" \
    -var="bucket_name=${R2_BUCKET_NAME:-masala-embed-models}"

echo "Applying Terraform plan"
terraform apply \
    -var="cloudflare_api_token=$CLOUDFLARE_API_TOKEN" \
    -var="account_id=$CLOUDFLARE_ACCOUNT_ID" \
    -var="bucket_name=${R2_BUCKET_NAME:-masala-embed-models}"
