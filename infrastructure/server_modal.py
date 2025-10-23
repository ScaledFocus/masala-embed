import csv
import os
import time

import faiss
import httpx
import modal
import numpy as np
import orjson
from fastapi import FastAPI, Response
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

APP_NAME = "masala-embed-server-modal"
VOLUME_NAME = "masala-embed-setup"

SETUP_DIR = "/vol/setup"
INDEX_PATH = f"{SETUP_DIR}/dish_index.faiss"
CSV_PATH = f"{SETUP_DIR}/dish_index.csv"

# Dependency image for the Modal app
image = modal.Image.debian_slim().pip_install(
    "fastapi",
    "httpx>=0.28.1",
    "orjson>=3.11.3",
    "faiss-cpu>=1.12.0",
    "numpy>=2.0.0",
)

# Modal app and volume
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


class EmbeddingRequest(BaseModel):
    input: str
    model: str


@app.function(
    image=image,
    volumes={"/vol": vol},
    timeout=600,
    # gpu="T4",
    env={
        "VLLM_URL": "https://scaledfocus--qwen3-embedding-inference-serve.modal.run/v1/embeddings"
    },
)
@modal.asgi_app()
@modal.concurrent(max_inputs=100)
def fastapi_app():
    api = FastAPI(default_response_class=ORJSONResponse)

    # State container for loaded resources
    state = {
        "faiss_index": None,
        "dishes": None,
        "query_buf": None,
        "http_client": None,
        "vllm_url": None,
    }

    @api.on_event("startup")
    async def startup_event():
        # Load FAISS index
        index = faiss.read_index(INDEX_PATH)
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = 32
        state["faiss_index"] = index

        # Load dishes (single-column CSV, no header)
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]
            dishes = [row[0] for row in rows]
        state["dishes"] = dishes

        # Preallocate query buffer
        dim = index.d
        state["query_buf"] = np.empty((1, dim), dtype="float32")

        # HTTP client and vLLM URL
        state["http_client"] = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=1.0, read=120.0, write=10.0, pool=120.0),
            headers={"Content-Type": "application/json"},
        )
        # Expect fully-qualified URL like: https://scaledfocus--qwen3-embedding-inference-serve.modal.run/v1/embeddings
        state["vllm_url"] = os.environ["VLLM_URL"]

    @api.on_event("shutdown")
    async def shutdown_event():
        client: httpx.AsyncClient = state["http_client"]
        await client.aclose()

    @api.post("/v1/embeddings")
    async def get_top_dish(req: EmbeddingRequest, response: Response):
        t0 = time.perf_counter()
        # Forward request to vLLM server
        client: httpx.AsyncClient = state["http_client"]
        vllm_url: str = state["vllm_url"]

        t_vllm_start = time.perf_counter()
        vllm_resp = await client.post(vllm_url, json=req.dict())
        # Include JSON parsing time in vLLM timing
        vllm_json = orjson.loads(vllm_resp.content)
        t_vllm_end = time.perf_counter()

        # Extract embedding
        embedding = vllm_json["data"][0]["embedding"]

        # Prepare normalized query in preallocated buffer
        qb = state["query_buf"]
        qb[0, :] = np.asarray(embedding, dtype="float32")

        t_norm_start = time.perf_counter()
        faiss.normalize_L2(qb)
        t_norm_end = time.perf_counter()

        # Search top-1
        index = state["faiss_index"]
        t_search_start = time.perf_counter()
        _, indices = index.search(qb, 1)
        t_search_end = time.perf_counter()
        top_idx = int(indices[0][0])

        # Map index to dish name
        dish_name = state["dishes"][top_idx]

        # Compute timings in milliseconds
        vllm_ms = (t_vllm_end - t_vllm_start) * 1000.0
        normalize_ms = (t_norm_end - t_norm_start) * 1000.0
        search_ms = (t_search_end - t_search_start) * 1000.0
        server_ms = (time.perf_counter() - t0) * 1000.0

        # Set timing headers for observability
        response.headers["X-Timing-vLLM"] = f"{vllm_ms:.3f}"
        response.headers["X-Timing-Normalize"] = f"{normalize_ms:.3f}"
        response.headers["X-Timing-Search"] = f"{search_ms:.3f}"
        response.headers["X-Timing-Server"] = f"{server_ms:.3f}"

        return {"dish": dish_name}

    return api
