import csv
from pathlib import Path

import faiss
import httpx
import numpy as np
import orjson
import uvicorn
from fastapi import FastAPI, Response
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
async def get_top_dish(req: EmbeddingRequest, response: Response):
    import time

    t0 = time.perf_counter()
    client: httpx.AsyncClient = app.state.http_client
    t_vllm_start = time.perf_counter()
    vllm_resp = await client.post(VLLM_EMBEDDINGS_PATH, json=req.dict())
    vllm_json = orjson.loads(vllm_resp.content)
    t_vllm_end = time.perf_counter()

    embedding = vllm_json["data"][0]["embedding"]

    qb = app.state.query_buf
    qb[0, :] = np.asarray(embedding, dtype="float32")

    t_norm_start = time.perf_counter()
    faiss.normalize_L2(qb)
    t_norm_end = time.perf_counter()

    index = app.state.faiss_index
    t_search_start = time.perf_counter()
    distances, indices = index.search(qb, 1)
    t_search_end = time.perf_counter()
    top_idx = int(indices[0][0])

    dishes: list[str] = app.state.dishes
    dish_name = dishes[top_idx]

    vllm_ms = (t_vllm_end - t_vllm_start) * 1000.0
    normalize_ms = (t_norm_end - t_norm_start) * 1000.0
    search_ms = (t_search_end - t_search_start) * 1000.0
    server_ms = (time.perf_counter() - t0) * 1000.0

    response.headers["X-Timing-vLLM"] = f"{vllm_ms:.3f}"
    response.headers["X-Timing-Normalize"] = f"{normalize_ms:.3f}"
    response.headers["X-Timing-Search"] = f"{search_ms:.3f}"
    response.headers["X-Timing-Server"] = f"{server_ms:.3f}"

    return {"dish": dish_name}


if __name__ == "__main__":
    uvicorn.run("server_local:app", host="0.0.0.0", port=8080, reload=False)
