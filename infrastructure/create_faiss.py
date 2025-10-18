import csv
from pathlib import Path

import faiss
import httpx
import numpy as np

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
VLLM_URL = "http://127.0.0.1:8000/v1/embeddings"
BATCH_SIZE = 256

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 32


def load_dishes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f]


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

    client = httpx.Client()

    # Get dimensions from first batch
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

    faiss.write_index(index, str(index_out_path))

    with dish_index_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for d in dishes:
            writer.writerow([d])

    client.close()


if __name__ == "__main__":
    main()
