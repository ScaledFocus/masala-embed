# Inference

## Modal

1. Sync dependencies using `uv`: `uv sync`
2. Setup Modal (if not already done): `uv run modal setup`
3. Deploy the app: `modal deploy vllm_inference_modal.py`

## Local

1. Sync dependencies using `uv`: `uv sync`
2. Start the vLLM server: `uv run vllm_inference_local.py`

## Usage Examples

### Single Sentence

```
curl -X POST "https://<modal_workspace|localhost>--qwen3-embedding-inference-serve.modal.run/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Need a cheesy, rainy-day dosa",
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }'
```

```
curl -X POST "http://0.0.0.0:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Need a cheesy, rainy-day dosa",
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }'
```

## List of Sentences

```
curl -X POST "https://<modal_workspace|localhost>--qwen3-embedding-inference-serve.modal.run/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Need a cheesy, rainy-day dosa",
      "Crispy dosa stuffed with spiced potatoes and melted cheese",
      "Fragrant biryani with tender lamb and aromatic basmati rice"
    ],
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }'
```
