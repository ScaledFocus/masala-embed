import csv
from pathlib import Path

import faiss
import httpx
import numpy as np

"""
Build FAISS index from embeddings fetched via local vLLM server.

- Reads dish names from dish_name.csv (one dish per line, no header)
- Queries local vLLM embeddings server in batches to get embeddings
- L2-normalizes all embeddings
- Builds an HNSW index with inner-product metric (cosine similarity)
- Writes index to dish_index.faiss
- Writes dish list to dish_index.csv (single column, no header)
"""

# Constants
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
VLLM_URL = "http://127.0.0.1:8000/v1/embeddings"
BATCH_SIZE = 256  # Tune based on GPU/CPU memory and vLLM server throughput

# HNSW parameters
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 32


def load_dishes(path: Path) -> list[str]:
    # dish_name.csv: one dish per line, no header
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                lines.append(name)
    return lines


def fetch_batch_embeddings(client: httpx.Client, texts: list[str]) -> np.ndarray:
    payload = {"input": texts, "model": MODEL_NAME}
    res = client.post(VLLM_URL, json=payload)
    data = res.json()["data"]
    # data is a list of objects each having an "embedding" list
    emb = np.asarray([d["embedding"] for d in data], dtype="float32")
    return emb


def main():
    base_dir = Path(__file__).resolve().parent
    dish_name_path = base_dir / "dish_name.csv"
    index_out_path = base_dir / "dish_index.faiss"
    dish_index_csv_path = base_dir / "dish_index.csv"

    dishes = load_dishes(dish_name_path)
    n = len(dishes)
    if n == 0:
        raise RuntimeError("No dishes found in dish_name.csv")

    # HTTP client with keep-alive
    client = httpx.Client()

    # First batch to determine embedding dimension
    first_batch = dishes[: min(BATCH_SIZE, n)]
    first_emb = fetch_batch_embeddings(client, first_batch)
    dim = first_emb.shape[1]

    # Preallocate full embedding matrix
    embeddings = np.empty((n, dim), dtype="float32")
    embeddings[: first_emb.shape[0], :] = first_emb

    # Process remaining batches
    offset = first_emb.shape[0]
    while offset < n:
        end = min(offset + BATCH_SIZE, n)
        batch = dishes[offset:end]
        emb = fetch_batch_embeddings(client, batch)
        embeddings[offset:end, :] = emb
        offset = end

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build HNSW index with inner product metric (equivalent to cosine on unit vectors)
    index = faiss.index_factory(dim, f"HNSW{HNSW_M}", faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(embeddings)

    # Persist index
    faiss.write_index(index, str(index_out_path))

    # Persist dish list as single-column CSV with no header
    with dish_index_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for d in dishes:
            writer.writerow([d])

    # Cleanup HTTP client
    client.close()


if __name__ == "__main__":
    main()
