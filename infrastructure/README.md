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

2. **Environment Variables**: Set the following in your `.env` file:
   ```bash
   CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
   R2_ACCESS_KEY_ID=your_r2_access_key
   R2_SECRET_ACCESS_KEY=your_r2_secret_key
   R2_BUCKET_NAME=masala-embed-models  # Optional, defaults to this
   ```

### Usage

```bash
# Set required parameters
export MODEL_NAME="Qwen/Qwen2.5-0.5B"
export QUANTIZATION_TYPE="Q4_0"

# Run quantization
./infrastructure/run_quantize.sh
```

## Model Inference Deployment

### Prerequisites

Same as quantization section above.

### Usage

Copy one of the example configurations and modify as needed:

```bash
# For Qwen (R2 model)
cp infrastructure/.env.qwen infrastructure/.env
# Edit .env with your actual credentials

# For ColQwen (HuggingFace model)
cp infrastructure/.env.colqwen infrastructure/.env

# For SigLIP (HuggingFace model)
cp infrastructure/.env.siglip infrastructure/.env
```

Deploy the model:

```bash
./infrastructure/run_inference.sh
```

### Configuration Files

- `.env.qwen` - Text embedding model from R2 (quantized)
- `.env.colqwen` - Multimodal model from HuggingFace
- `.env.siglip` - Multimodal model from HuggingFace

# API Usage

## Endpoints

### Health Check

```bash
GET /health
```

### Generate Embeddings

```bash
POST /embed
```

**Request:**

```json
{
  "text": "Your text to embed (optional)",
  "image_url": "https://example.com/image.jpg (optional)"
}
```

**Response:**

```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "model": "model_name"
}
```

## Examples

```bash
# Text embedding
curl -X POST https://your-modal-endpoint/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "A beautiful sunset"}'

# Image embedding
curl -X POST https://your-modal-endpoint/embed \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'
```
