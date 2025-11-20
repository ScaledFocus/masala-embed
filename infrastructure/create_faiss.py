import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from PIL import Image
from vllm import LLM

# Configuration
MODEL = os.getenv("MODEL", "google/siglip2-base-patch16-224")
DATASET_CSV = os.getenv("DATASET_CSV", "dataset.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "32"))


def embed_multimodal(llm: LLM, text: str | None, image_path: str | None) -> np.ndarray:
    """Embed text and/or image, matching server/evaluation behavior."""
    parts = []

    if text and text.strip():
        out = llm.embed([text])
        text_embed = np.asarray(out[0].outputs.embedding, dtype="float32")
        text_embed = text_embed / np.linalg.norm(text_embed)
        parts.append(text_embed)

    if image_path and image_path.strip():
        img = Image.open(image_path).convert("RGB")
        out = llm.embed([img])
        image_embed = np.asarray(out[0].outputs.embedding, dtype="float32")
        image_embed = image_embed / np.linalg.norm(image_embed)
        parts.append(image_embed)

    if not parts:
        raise ValueError("Must provide at least text or image")

    if len(parts) == 1:
        return parts[0]
    else:
        # Average normalized embeddings, then normalize again
        combined = (parts[0] + parts[1]) / 2.0
        combined = combined / np.linalg.norm(combined)
        return combined


def main():
    base = Path(__file__).resolve().parent
    dataset_path = base / DATASET_CSV
    index_out_path = base / "setup" / "dish_index.faiss"
    dish_index_csv_path = base / "setup" / "dish_index.csv"

    # Create setup directory
    index_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Get unique dishes (one embedding per dish)
    print(f"Found {len(df)} total rows")
    unique_dishes = df["dish_name"].unique()
    print(f"Found {len(unique_dishes)} unique dishes")

    # Initialize model
    print(f"Loading model: {MODEL}")
    llm = LLM(
        model=MODEL,
        task="embed",
        tokenizer_mode="auto",
        gpu_memory_utilization=0.5,
    )

    # Process dishes and create embeddings
    dish_names = []
    embeddings = []

    for dish_name in unique_dishes:
        # Get first row for this dish
        dish_row = df[df["dish_name"] == dish_name].iloc[0]

        text = str(dish_row["text"]).strip() if pd.notna(dish_row["text"]) else None
        image = str(dish_row["image"]).strip() if pd.notna(dish_row["image"]) else None

        if not text and not image:
            print(f"Warning: Skipping dish '{dish_name}' - no text or image")
            continue

        try:
            embedding = embed_multimodal(llm, text, image)
            embeddings.append(embedding)
            dish_names.append(dish_name)
            print(f"Processed: {dish_name}")
        except Exception as e:
            print(f"Error processing dish '{dish_name}': {e}")
            continue

    if not embeddings:
        raise RuntimeError("No valid dish embeddings created")

    # Stack embeddings
    embeddings_matrix = np.vstack(embeddings)  # (n_dishes, dim)
    dim = embeddings_matrix.shape[1]

    print(f"\nCreating FAISS index with {len(embeddings)} dishes, dim={dim}")

    # Create HNSW index for approximate nearest neighbor search
    index = faiss.index_factory(dim, f"HNSW{HNSW_M}", faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    # Add embeddings to index (already normalized)
    index.add(embeddings_matrix)

    # Save FAISS index
    faiss.write_index(index, str(index_out_path))
    print(f"FAISS index saved to: {index_out_path}")

    # Save dish names (in same order as FAISS index)
    with open(dish_index_csv_path, "w", encoding="utf-8") as f:
        for dish_name in dish_names:
            f.write(f"{dish_name}\n")
    print(f"Dish names saved to: {dish_index_csv_path}")

    print(f"\nDone! Created index with {len(dish_names)} dishes")


if __name__ == "__main__":
    main()
