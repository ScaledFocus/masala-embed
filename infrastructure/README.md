# Infrastructure

## vLLM Installation

Until vLLM stable supports SigLIP 2, install as follows:

```bash
echo xformers > exclude.txt
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly --excludes exclude.txt
```

## Create FAISS Index

Build the FAISS index from your dataset with multimodal embeddings.

```bash
MODEL=google/siglip2-base-patch16-224
DATASET_CSV=dataset.csv
BATCH_SIZE=32
HNSW_M=32
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=32
python infrastructure/create_faiss.py
```

### Environment Variables

1. MODEL: Model to use for embeddings (default: `google/siglip2-base-patch16-224`).
2. DATASET_CSV: Path to dataset CSV file (default: `dataset.csv`).
3. BATCH_SIZE: Batch size for processing (default: `32`).
4. HNSW_M: HNSW M parameter (default: `32`).
5. HNSW_EF_CONSTRUCTION: HNSW efConstruction parameter (default: `200`).
6. HNSW_EF_SEARCH: HNSW efSearch parameter (default: `32`).

### Output

- `infrastructure/setup/dish_index.faiss` - FAISS index file
- `infrastructure/setup/dish_index.csv` - Dish names (same order as index)

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
    "dish": "Biryani"
}
```

You can provide:

- Both `text` and `image` (multimodal)
- Only `text`
- Only `image`

At least one must be provided.
