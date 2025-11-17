import shutil
from pathlib import Path

import faiss
import numpy as np
from vllm import LLM

MODEL = "openai/clip-vit-base-patch32"
# MODEL = "google/siglip-base-patch16-224"
BATCH_SIZE = 512

# HNSW params (good defaults; tune for your dataset/latency)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 32


def iter_lines(path: Path, batch_size: int):
    batch = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            batch.append(s)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def embed_batch(llm: LLM, texts: list[str]) -> np.ndarray:
    out = llm.embed(texts)  # same API as server.py
    # out[i].outputs.embedding -> list[float]
    embs = [np.asarray(o.outputs.embedding, dtype="float32") for o in out]
    return np.stack(embs, axis=0)  # (batch, dim)


def main():
    base = Path(__file__).resolve().parent
    dish_name_path = base / "setup" / "dish_name.csv"
    index_out_path = base / "setup" / "dish_index.faiss"
    dish_index_csv_path = base / "setup" / "dish_index.csv"

    # 0) Make sure the server will read the same order
    # If dish_name.csv already has one name per line, just copy it.
    shutil.copyfile(dish_name_path, dish_index_csv_path)

    # 1) Init model
    llm = LLM(
        model=MODEL,
        task="embed",
        tokenizer_mode="auto",
        gpu_memory_utilization=0.5,
    )

    # 2) Prime on first batch to get dimension and build index
    gen = iter_lines(dish_name_path, BATCH_SIZE)
    first_batch = next(gen, None)
    if not first_batch:
        raise RuntimeError("No dishes found in dish_name.csv")

    first_emb = embed_batch(llm, first_batch)  # (b, dim)
    dim = first_emb.shape[1]

    # Normalize (cosine via inner product)
    faiss.normalize_L2(first_emb)

    # HNSW index
    index = faiss.index_factory(dim, f"HNSW{HNSW_M}", faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    # Add first batch
    index.add(first_emb)

    # 3) Stream remaining batches: embed -> normalize -> add
    for batch in gen:
        emb = embed_batch(llm, batch)
        faiss.normalize_L2(emb)
        index.add(emb)

    # 4) Persist index
    faiss.write_index(index, str(index_out_path))


if __name__ == "__main__":
    main()
