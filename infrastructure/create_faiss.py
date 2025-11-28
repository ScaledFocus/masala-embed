import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from PIL import Image
from vllm import LLM

MODEL = os.getenv("MODEL", "google/siglip2-base-patch16-224")
DATASET_CSV = os.getenv("DATASET_CSV", "dataset.csv")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.5"))


def embed_text(llm: LLM, text: str) -> np.ndarray:
    """Embed text and return normalized embedding."""
    out = llm.embed([text])
    arr = np.asarray(out[0].outputs.embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(arr)
    return arr.flatten()


def embed_image(llm: LLM, image_path: str) -> np.ndarray:
    """Embed image and return normalized embedding."""
    img = Image.open(image_path).convert("RGB")
    out = llm.embed([img])
    arr = np.asarray(out[0].outputs.embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(arr)
    return arr.flatten()


def main():
    base = Path(__file__).resolve().parent
    dataset_path = base / DATASET_CSV
    text_index_path = base / "setup" / "dish_index_text.faiss"
    image_index_path = base / "setup" / "dish_index_image.faiss"
    dish_index_csv_path = base / "setup" / "dish_index.csv"

    text_index_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    print(f"Found {len(df)} total rows")
    unique_dishes = df["dish_name"].unique()
    print(f"Found {len(unique_dishes)} unique dishes")

    print(f"Loading model: {MODEL}")
    llm = LLM(
        model=MODEL,
        task="embed",
        tokenizer_mode="auto",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    dish_names = []
    text_embeddings = []
    image_embeddings = []
    dim = None

    # First pass: collect all embeddings
    for dish_name in unique_dishes:
        dish_row = df[df["dish_name"] == dish_name].iloc[0]

        text = str(dish_row["text"]).strip() if pd.notna(dish_row["text"]) else None
        image = str(dish_row["image"]).strip() if pd.notna(dish_row["image"]) else None

        if not text and not image:
            print(f"Warning: Skipping dish '{dish_name}' - no text or image")
            continue

        try:
            # Get embeddings for available modalities
            if text and text.strip():
                text_embed = embed_text(llm, text)
                if dim is None:
                    dim = len(text_embed)
            else:
                text_embed = None

            if image and image.strip():
                image_embed = embed_image(llm, image)
                if dim is None:
                    dim = len(image_embed)
            else:
                image_embed = None

            # Add to lists (use zero vector for missing modality)
            if text_embed is not None:
                text_embeddings.append(text_embed)
            else:
                text_embeddings.append(np.zeros(dim, dtype="float32"))

            if image_embed is not None:
                image_embeddings.append(image_embed)
            else:
                image_embeddings.append(np.zeros(dim, dtype="float32"))

            dish_names.append(dish_name)
            print(f"Processed: {dish_name}")
        except Exception as e:
            print(f"Error processing dish '{dish_name}': {e}")
            continue

    if not dish_names:
        raise RuntimeError("No valid dish embeddings created")

    # Stack embeddings into matrices
    text_matrix = np.vstack(text_embeddings)
    image_matrix = np.vstack(image_embeddings)

    print(f"\nCreating FAISS indices with {len(dish_names)} dishes, dim={dim}")

    text_index = faiss.IndexFlatIP(dim)
    text_index.add(text_matrix)

    image_index = faiss.IndexFlatIP(dim)
    image_index.add(image_matrix)

    faiss.write_index(text_index, str(text_index_path))
    faiss.write_index(image_index, str(image_index_path))
    print(f"Text FAISS index saved to: {text_index_path}")
    print(f"Image FAISS index saved to: {image_index_path}")

    with open(dish_index_csv_path, "w", encoding="utf-8") as f:
        for dish_name in dish_names:
            f.write(f"{dish_name}\n")
    print(f"Dish names saved to: {dish_index_csv_path}")

    print(f"\nDone! Created IndexFlatIP indices with {len(dish_names)} dishes")


if __name__ == "__main__":
    main()
