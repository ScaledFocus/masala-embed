# Setup Instructions

## Loading environment variables

```
set -a
source .env
set +a
```

## Cloudflare

1. `CLOUDFLARE_ACCOUNT_ID` environment variable required.
2. Install `wrangler`.
3. Log into wrangler.
4. Run `run_terraform.sh`

This will setup R2 bucket to store models.

## Model Quantization

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com) and install the CLI:

   ```bash
   modal setup
   ```

   You would need to add these environment variables to your Modal secrets:

2. **Environment Variables**: Set the following in your `.env` file:
   ```bash
   CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
   R2_ACCESS_KEY_ID=your_r2_access_key
   R2_SECRET_ACCESS_KEY=your_r2_secret_key
   R2_BUCKET_NAME=masala-embed-models  # Optional, defaults to this
   ```

### Usage

#### Usage

```bash
# Set required parameters
export MODEL_NAME="Qwen/Qwen2.5-0.5B"
export QUANTIZATION_TYPE="Q4_0"

# Run quantization
./infrastructure/run_quantize.sh
```

# API Usage

The Masala Embed API provides 4 endpoints for health checking and generating embeddings using different models.

## Base URL

```
https://your-worker-domain.workers.dev
```

## Endpoints

### Health Check

Check if all embedding services are healthy.

```bash
GET /health
```

**Response:**

```json
{
  "siglip": true,
  "colqwen": true,
  "qwen": false
}
```

### Generate Embeddings

All embedding endpoints accept the same request format and return embeddings in a consistent format.

#### SigLIP Embeddings

```bash
POST /embedding/siglip
```

#### ColQwen Embeddings

```bash
POST /embedding/colqwen
```

#### Qwen Embeddings

```bash
POST /embedding/qwen
```

**Request Format:**

```json
{
  "text": "Your text to embed (optional)",
  "image_url": "https://example.com/image.jpg (optional)"
}
```

**Response Format:**

```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "model": "siglip"
}
```

## Examples

### Text Embedding

```bash
curl -X POST https://your-worker-domain.workers.dev/embedding/siglip \
  -H "Content-Type: application/json" \
  -d '{"text": "A beautiful sunset over the mountains"}'
```

### Image Embedding

```bash
curl -X POST https://your-worker-domain.workers.dev/embedding/colqwen \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/sunset.jpg"}'
```

### Combined Text and Image

```bash
curl -X POST https://your-worker-domain.workers.dev/embedding/qwen \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A beautiful sunset",
    "image_url": "https://example.com/sunset.jpg"
  }'
```

### Health Check

```bash
curl https://your-worker-domain.workers.dev/health
```

## Error Responses

If a request fails, you'll receive an error response:

```json
{
  "error": "Request failed"
}
```
