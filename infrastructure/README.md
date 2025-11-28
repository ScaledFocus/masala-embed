# Infrastructure

## Quick Start (Modal - Recommended)

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com) and install the CLI:
   ```bash
   modal setup
   ```

2. **Environment Variables**: Copy and configure:
   ```bash
   # For Qwen (text embedding)
   cp infrastructure/.env.qwen infrastructure/.env

   # For ColQwen (multimodal)
   cp infrastructure/.env.colqwen infrastructure/.env

   # For SigLIP (multimodal)
   cp infrastructure/.env.siglip infrastructure/.env
   ```

### Deploy

```bash
./infrastructure/run_inference.sh
```

---

## Local Setup (GPU)

NOTE: `cd` into infrastructure first

1. Install dependencies
   ```bash
   uv sync
   ```

2. Prepare dish names
   - Ensure `infrastructure/setup/dish_name.csv` exists (single column, no header)

3. Start the local vLLM embedding server
   ```bash
   uv run vllm_inference_local.py
   # Exposes: http://127.0.0.1:8000/v1/embeddings
   ```

4. Build the FAISS index
   ```bash
   uv run create_faiss.py
   # Writes dish_index.faiss and dish_index.csv
   ```

5. Start the local FastAPI server
   ```bash
   uv run server_local.py
   ```

6. Test locally
   ```bash
   curl -X POST "http://0.0.0.0:8080/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "input": "Need a cheesy dish",
       "model": "Qwen/Qwen3-Embedding-0.6B"
     }'
   ```

---

## Modal Setup (Full)

### Prerequisites

- Modal CLI installed and authenticated: `uv run modal setup`
- vLLM embedding server deployed (see below)

### 1. Deploy vLLM Embedding Server

```bash
modal deploy vllm_inference_modal.py
# Note the endpoint URL
```

### 2. Upload artifacts to Modal Volume

```bash
# Upload dish names
modal run create_embeddings_modal::main

# Upload FAISS index
modal run create_faiss_modal::main
```

### 3. Deploy FastAPI Server

```bash
modal deploy server_modal.py
```

### 4. Test

```bash
curl -X POST "https://scaledfocus--masala-embed-server-modal-fastapi-app.modal.run/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Need some italian dish",
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }'
```

---

## API Reference

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

---

## Configuration Files

| File | Description |
|------|-------------|
| `.env.qwen` | Text embedding model (quantized) |
| `.env.colqwen` | Multimodal model from HuggingFace |
| `.env.siglip` | Multimodal model from HuggingFace |

---

## Cloudflare R2 Setup (Optional)

For model storage in Cloudflare R2:

1. Set environment variables:
   ```bash
   CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
   R2_ACCESS_KEY_ID=your_r2_access_key
   R2_SECRET_ACCESS_KEY=your_r2_secret_key
   R2_BUCKET_NAME=masala-embed-models
   ```

2. Run terraform:
   ```bash
   ./infrastructure/run_terraform.sh
   ```

3. Quantize and upload model:
   ```bash
   export MODEL_NAME="Qwen/Qwen2.5-0.5B"
   export QUANTIZATION_TYPE="Q4_0"
   ./infrastructure/run_quantize.sh
   ```
