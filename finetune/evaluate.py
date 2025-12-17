import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoProcessor

# Configuration from environment variables
# Default MODEL_PATH is the original SigLIP model for comparison
MODEL_PATH = os.getenv("MODEL_PATH")
print(f"MODEL_PATH: {MODEL_PATH}")
BENCHMARK_CSV = os.getenv("BENCHMARK_CSV", "test.csv")
OUTPUT_CSV_BASE = os.getenv("OUTPUT_CSV", "evaluation_results.csv")

# Add timestamp to output filename if not explicitly provided with timestamp
if OUTPUT_CSV_BASE == "evaluation_results.csv":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, ext = os.path.splitext(OUTPUT_CSV_BASE)
    OUTPUT_CSV = f"{base_name}_{timestamp}{ext}"
else:
    OUTPUT_CSV = OUTPUT_CSV_BASE


def load_model_and_processor(model_path: str, device: torch.device):
    """Load the (possibly finetuned) SigLIP model and processor."""
    print(f"Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


def get_text_embedding(
    model: AutoModel, processor: AutoProcessor, text: str, device: torch.device
) -> np.ndarray:
    """Get normalized text embedding using the text tower only."""
    if not text:
        raise ValueError("Empty text provided to get_text_embedding")

    with torch.no_grad():
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Use the dedicated text tower to avoid needing pixel_values
        if hasattr(model, "get_text_features"):
            embeds = model.get_text_features(**inputs)  # (batch, dim)
        else:
            outputs = model(**inputs)
            embeds = outputs.text_embeds  # Fallback for compatible models

        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        vec = embeds[0].detach().cpu().numpy().astype("float32")
    return vec[None, :]  # Shape: (1, dim)


def get_image_embedding(
    model: AutoModel, processor: AutoProcessor, image_path: str, device: torch.device
) -> np.ndarray | None:
    """Get normalized image embedding using the vision tower only.

    Returns None if the image cannot be loaded or is corrupted.
    """
    if not image_path:
        raise ValueError("Empty image path provided to get_image_embedding")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: failed to open image {image_path}: {e}")
        return None

    with torch.no_grad():
        inputs = processor(
            images=[image],
            return_tensors="pt",
        ).to(device)

        # Use the dedicated vision tower to avoid requiring text inputs
        if hasattr(model, "get_image_features"):
            embeds = model.get_image_features(**inputs)  # (batch, dim)
        else:
            outputs = model(**inputs)
            embeds = outputs.image_embeds  # Fallback for compatible models

        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        vec = embeds[0].detach().cpu().numpy().astype("float32")
    return vec[None, :]  # Shape: (1, dim)


def compute_ndcg(relevance_scores, k):
    """Compute NDCG@k for a single query."""
    if len(relevance_scores) == 0:
        return 0.0

    # Get top-k scores
    relevance_scores = np.array(relevance_scores[:k])

    # DCG: sum of rel_i / log2(i + 1)
    dcg = np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))

    # IDCG: DCG of perfect ranking
    ideal_scores = np.sort(relevance_scores)[::-1]
    idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_model(model_path, benchmark_csv, output_csv, _gpu_memory_utilization=None):
    """Evaluate model on benchmark dataset using the transformers SigLIP model."""
    print(f"Model: {model_path}")
    print(f"Dataset: {benchmark_csv}")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(model_path, device)

    # Load dataset
    df = pd.read_csv(benchmark_csv)

    # Build dish embeddings (text and image separately)
    dish_text_embeddings = {}
    dish_image_embeddings = {}

    print("Computing dish embeddings...")
    for dish_name in df["dish_name"].unique():
        dish_rows = df[df["dish_name"] == dish_name]
        dish_row = dish_rows.iloc[0]

        # Extract text and image, handling possible NaN/empty values
        dish_text = (
            str(dish_row["text"]).strip() if pd.notna(dish_row["text"]) else None
        )
        dish_image = (
            str(dish_row["image"]).strip() if pd.notna(dish_row["image"]) else None
        )

        if dish_text:
            dish_text_embeddings[dish_name] = get_text_embedding(
                model, processor, dish_text, device
            )

        if dish_image:
            img_emb = get_image_embedding(model, processor, dish_image, device)
            if img_emb is not None:
                dish_image_embeddings[dish_name] = img_emb

    dish_names = list(
        set(dish_text_embeddings.keys()) | set(dish_image_embeddings.keys())
    )

    # Create matrices for vectorized search
    dish_text_matrix = None
    dish_image_matrix = None

    if dish_text_embeddings:
        # Get dimension from first embedding
        first_text_embed = list(dish_text_embeddings.values())[0]
        text_dim = first_text_embed.shape[1]

        text_embeds_list = []
        for name in dish_names:
            if name in dish_text_embeddings:
                text_embeds_list.append(dish_text_embeddings[name])
            else:
                text_embeds_list.append(np.zeros((1, text_dim)))
        dish_text_matrix = np.vstack(text_embeds_list)

    if dish_image_embeddings:
        # Get dimension from first embedding
        first_image_embed = list(dish_image_embeddings.values())[0]
        image_dim = first_image_embed.shape[1]

        image_embeds_list = []
        for name in dish_names:
            if name in dish_image_embeddings:
                image_embeds_list.append(dish_image_embeddings[name])
            else:
                image_embeds_list.append(np.zeros((1, image_dim)))
        dish_image_matrix = np.vstack(image_embeds_list)

    recall_at_1 = 0
    recall_at_5 = 0
    ndcg_scores = []
    total_queries = 0

    print("Evaluating queries...")
    for idx, row in df.iterrows():
        query_text = str(row["text"]).strip() if pd.notna(row["text"]) else None
        query_image = str(row["image"]).strip() if pd.notna(row["image"]) else None
        true_dish = str(row["dish_name"])

        # Skip if both text and image are missing
        if not query_text and not query_image:
            continue

        # Run separate searches and combine scores
        combined_scores = {}

        if query_text and dish_text_matrix is not None:
            query_text_embed = get_text_embedding(
                model, processor, query_text, device
            )
            text_similarities = cosine_similarity(query_text_embed, dish_text_matrix)[0]
            for dish_name, score in zip(dish_names, text_similarities):
                combined_scores[dish_name] = combined_scores.get(
                    dish_name, 0.0
                ) + float(score)

        if query_image and dish_image_matrix is not None:
            query_image_embed = get_image_embedding(
                model, processor, query_image, device
            )
            if query_image_embed is not None:
                image_similarities = cosine_similarity(
                    query_image_embed, dish_image_matrix
                )[0]
                for dish_name, score in zip(dish_names, image_similarities):
                    combined_scores[dish_name] = combined_scores.get(
                        dish_name, 0.0
                    ) + float(score)

        # Sort by combined score
        sorted_dishes = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_dishes_sorted = [dish for dish, score in sorted_dishes]

        # Recall@1
        if top_dishes_sorted and top_dishes_sorted[0] == true_dish:
            recall_at_1 += 1

        # Recall@5
        top_5_dishes = top_dishes_sorted[:5]
        if true_dish in top_5_dishes:
            recall_at_5 += 1

        # NDCG@5: binary relevance (1 if correct dish, 0 otherwise)
        relevance = [1.0 if dish == true_dish else 0.0 for dish in top_dishes_sorted]
        ndcg_scores.append(compute_ndcg(relevance, k=5))

        total_queries += 1

    r_at_1 = recall_at_1 / total_queries if total_queries > 0 else 0.0
    r_at_5 = recall_at_5 / total_queries if total_queries > 0 else 0.0
    ndcg_at_5 = np.mean(ndcg_scores) if ndcg_scores else 0.0

    results_df = pd.DataFrame(
        [
            {
                "model_path": model_path,
                "benchmark_csv": benchmark_csv,
                "R@1": r_at_1,
                "R@5": r_at_5,
                "NDCG@5": ndcg_at_5,
                "total_queries": total_queries,
                "total_dishes": len(dish_names),
            }
        ]
    )

    if os.path.exists(output_csv):
        results_df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        results_df.to_csv(output_csv, index=False)

    print(f"\nResults saved to {output_csv}")
    print(f"R@1: {r_at_1:.4f}, R@5: {r_at_5:.4f}, NDCG@5: {ndcg_at_5:.4f}")

    return results_df.iloc[0].to_dict()


if __name__ == "__main__":
    results = evaluate_model(
        MODEL_PATH, BENCHMARK_CSV, OUTPUT_CSV, None
    )
