#!/bin/bash

set -e

set -a
source .env
set +a

echo "Checking if required environment variables are set."
if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
    echo "Error: CLOUDFLARE_ACCOUNT_ID not set"
    exit 1
fi

if [ -z "$CLOUDFLARE_API_KEY" ]; then
    echo "Error: CLOUDFLARE_ACCOUNT_ID not set"
    exit 1
fi

cd terraform

echo "Initializing Terraform"
terraform init

echo "Planning Terraform deployment"
terraform plan \
    -var="cloudflare_account_id=$CLOUDFLARE_ACCOUNT_ID" \
    -var="cloudflare_api_key=$CLOUDFLARE_API_KEY" \
    -var="bucket_name=${R2_BUCKET_NAME:-masala-embed-models}" \
    -out=plan.out

echo "Applying Terraform plan"
terraform apply plan.out
