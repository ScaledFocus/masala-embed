import csv
from pathlib import Path

import faiss
import httpx
import numpy as np
import orjson
import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    input: str
    model: str


VLLM_BASE_URL = (
    "http://127.0.0.1:8000"  # vLLM server started by vllm_inference_local.py
)
VLLM_EMBEDDINGS_PATH = "/v1/embeddings"

app = FastAPI(default_response_class=ORJSONResponse)


@app.on_event("startup")
async def startup_event():
    base_dir = Path(__file__).resolve().parent
    setup_dir = base_dir / "setup"
    index_path = setup_dir / "dish_index.faiss"
    csv_path = setup_dir / "dish_index.csv"

    app.state.faiss_index = faiss.read_index(str(index_path))
    index = app.state.faiss_index
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = 32

    # Single-column CSV with no header
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
        app.state.dishes = [row[0] for row in rows]

    # Preallocate reusable query buffer
    dim = index.d
    app.state.query_buf = np.empty((1, dim), dtype="float32")

    # Reuse a single async HTTP client with connection pooling
    app.state.http_client = httpx.AsyncClient(
        base_url=VLLM_BASE_URL,
        timeout=httpx.Timeout(connect=1.0, read=120.0, write=10.0, pool=120.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        headers={"Content-Type": "application/json"},
    )


@app.on_event("shutdown")
async def shutdown_event():
    # Close the HTTP client cleanly
    client: httpx.AsyncClient = app.state.http_client
    await client.aclose()


@app.post("/v1/embeddings")
async def get_top_dish(req: EmbeddingRequest):
    # Forward request to local vLLM embedding server
    client: httpx.AsyncClient = app.state.http_client
    vllm_resp = await client.post(VLLM_EMBEDDINGS_PATH, json=req.dict())
    vllm_json = orjson.loads(vllm_resp.content)

    # Extract single embedding
    embedding = vllm_json["data"][0]["embedding"]

    # Prepare normalized query in preallocated buffer
    qb = app.state.query_buf
    qb[0, :] = np.asarray(embedding, dtype="float32")
    faiss.normalize_L2(qb)

    # Nearest neighbor search (k=1)
    index = app.state.faiss_index
    distances, indices = index.search(qb, 1)
    top_idx = int(indices[0][0])

    # Map to dish name
    dishes: list[str] = app.state.dishes
    dish_name = dishes[top_idx]

    return {"dish": dish_name}


if __name__ == "__main__":
    uvicorn.run("server_local:app", host="0.0.0.0", port=8080, reload=False)
