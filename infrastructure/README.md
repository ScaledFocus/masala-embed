# Infrastructure

Supports both local and Modal deployments.

## Local Setup (GPU)

NOTE: `cd` into infrastructure first

1. Install dependencies

- uv sync

2. Prepare dish names

- Ensure the file infrastructure/setup/dish_name.csv exists (single column, no header).

3. Start the local vLLM embedding server

- From the infrastructure directory:
    - uv run vllm_inference_local.py
- This exposes embeddings at: http://127.0.0.1:8000/v1/embeddings

4. Build the FAISS index using vLLM embeddings

- From the infrastructure directory:
    - uv run create_faiss.py
- This writes dish_index.faiss and dish_index.csv to the setup in Infrastructure.

5. Start the local FastAPI server

- From the infrastructure directory:
    - uv run server_local.py

6. Test locally

```
curl -X POST "http://0.0.0.0:8080/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Need an cheesy dish",
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }'
```

## Modal Setup

Prereqs

- You must already have your vLLM embedding server deployed on Modal (as per the existing vllm_inference_modal.py in your workspace). Note the full embeddings URL, for example:
    - https://scaledfocus--qwen3-embedding-inference-serve.modal.run/v1/embeddings

1. Install dependencies and Modal CLI

- uv sync
- Initialize Modal in your environment (one-time):
    - uv run modal setup

2. Upload artifacts to a Modal Volume

- The server expects the following in the Modal Volume named masala-embed-setup at /setup:
    - /setup/dish_index.faiss
    - /setup/dish_index.csv
- From the infrastructure directory, after generating the files locally:
    - modal run create_embeddings_modal::main
        - Uploads setup/dish_index.csv to the Volume
    - modal run create_faiss_modal::main
        - Uploads setup/dish_index.faiss to the Volume

3. Deploy the FastAPI server on Modal

- The server needs the vLLM embeddings URL via environment variable VLLM_URL.
- Deploy:
    - modal deploy server_modal.py

4. Test on Modal

```
curl -X POST "https://scaledfocus--masala-embed-server-modal-fastapi-app.modal.run/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Need some italian dish",
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }'
```
