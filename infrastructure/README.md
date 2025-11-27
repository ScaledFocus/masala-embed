# Infrastructure

## vLLM Installation

Until vLLM stable supports SigLIP 2, install as follows:

```bash
echo xformers > exclude.txt
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly --excludes exclude.txt
```

## Create FAISS Index

Build dual FAISS indices (text and image) from your dataset.

```bash
MODEL=google/siglip2-base-patch16-224
DATASET_CSV=dataset.csv
GPU_MEMORY_UTILIZATION=0.5
python infrastructure/create_faiss.py
```

### Environment Variables

1. MODEL: Model to use for embeddings (default: `google/siglip2-base-patch16-224`).
2. DATASET_CSV: Path to dataset CSV file (default: `dataset.csv`).
3. GPU_MEMORY_UTILIZATION: GPU memory for vLLM (default: `0.5`).

### Output

- `infrastructure/setup/dish_index_text.faiss` - IndexFlatIP for text embeddings
- `infrastructure/setup/dish_index_image.faiss` - IndexFlatIP for image embeddings
- `infrastructure/setup/dish_index.csv` - Dish names (same order as both indices)

## Run Server

Start the FastAPI server for dish retrieval.

```bash
python infrastructure/server.py
```

### API Endpoint

**POST** `/v1/dish`

Request:

```json
{
    "text": "spicy rice dish",
    "image": "https://example.com/dish.jpg"
}
```

Response:

```json
{
    "dishes": ["Biryani", "Pulao", "Fried Rice", "Khichdi", "Tehri"]
}
```

You can provide:

- Both `text` and `image` - runs separate FAISS searches and combines top 5 results
- Only `text` - searches FAISS text index
- Only `image` - searches FAISS image index

At least one must be provided.

**How it works:**

1. Text provided → text embedding → FAISS text index search → top 5 results
2. Image provided → image embedding → FAISS image index search → top 5 results
3. Both provided → combines scores from both searches → returns top 5

### Environment Variables

1. MODEL: Model to use (default: `google/siglip2-base-patch16-224`).
2. GPU_MEMORY_UTILIZATION: GPU memory for vLLM (default: `0.8`).
3. TOP_K: Number of results to return (default: `5`).

Run with:

```bash
GPU_MEMORY_UTILIZATION=0.8 \
TOP_K=5 \
uvicorn infrastructure.server:app --host 0.0.0.0 --port 8000 --loop uvloop --http httptools
```
