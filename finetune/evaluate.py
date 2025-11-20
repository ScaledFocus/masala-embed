import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoProcessor

MODEL_PATH = os.getenv("MODEL_PATH", "google/siglip2-base-patch16-224")
BENCHMARK_CSV = os.getenv("BENCHMARK_CSV", "test.csv")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "evaluation_results.csv")


def get_embeddings(model, processor, device, text=None, image_path=None):
    """Get embeddings for text and/or image with flexible modality support."""
    with torch.no_grad():
        # Handle empty/None text
        has_text = text is not None and str(text).strip()
        has_image = image_path is not None and str(image_path).strip()

        if has_text and has_image:
            # Both text and image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            text_embed = outputs.text_embeds.cpu().numpy()
            image_embed = outputs.image_embeds.cpu().numpy()
            text_embed = text_embed / np.linalg.norm(text_embed, axis=1, keepdims=True)
            image_embed = image_embed / np.linalg.norm(image_embed, axis=1, keepdims=True)
            combined_embed = (text_embed + image_embed) / 2.0
            combined_embed = combined_embed / np.linalg.norm(combined_embed, axis=1, keepdims=True)
            return combined_embed
        elif has_text:
            # Text only
            inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            text_embed = outputs.text_embeds.cpu().numpy()
            text_embed = text_embed / np.linalg.norm(text_embed, axis=1, keepdims=True)
            return text_embed
        elif has_image:
            # Image only
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            image_embed = outputs.image_embeds.cpu().numpy()
            image_embed = image_embed / np.linalg.norm(image_embed, axis=1, keepdims=True)
            return image_embed
        else:
            raise ValueError("Either text or image_path must be provided")


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


def evaluate_model(model_path, benchmark_csv, device, output_csv):
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    df = pd.read_csv(benchmark_csv)

    dish_embeddings = {}
    for dish_name in df["dish_name"].unique():
        dish_rows = df[df["dish_name"] == dish_name]
        dish_row = dish_rows.iloc[0]

        # Extract text and image, handling possible NaN/empty values
        dish_text = str(dish_row["text"]).strip() if pd.notna(dish_row["text"]) else None
        dish_image = str(dish_row["image"]).strip() if pd.notna(dish_row["image"]) else None

        dish_embeddings[dish_name] = get_embeddings(
            model, processor, device, text=dish_text, image_path=dish_image
        )

    dish_names = list(dish_embeddings.keys())
    dish_embeds_matrix = np.vstack([dish_embeddings[name] for name in dish_names])

    recall_at_1 = 0
    recall_at_5 = 0
    ndcg_scores = []
    total_queries = 0

    for idx, row in df.iterrows():
        # Extract text and image, handling possible NaN/empty values
        query_text = str(row["text"]).strip() if pd.notna(row["text"]) else None
        query_image = str(row["image"]).strip() if pd.notna(row["image"]) else None
        true_dish = str(row["dish_name"])

        # Skip if both text and image are missing
        if not query_text and not query_image:
            continue

        query_embed = get_embeddings(
            model, processor, device, text=query_text, image_path=query_image
        )

        similarities = cosine_similarity(query_embed, dish_embeds_matrix)[0]
        top_indices = np.argsort(similarities)[::-1]

        # Recall@1
        top_1_dish = dish_names[top_indices[0]]
        if top_1_dish == true_dish:
            recall_at_1 += 1

        # Recall@5
        top_5_dishes = [dish_names[i] for i in top_indices[:5]]
        if true_dish in top_5_dishes:
            recall_at_5 += 1

        # NDCG@5: binary relevance (1 if correct dish, 0 otherwise)
        relevance = [1.0 if dish_names[i] == true_dish else 0.0 for i in top_indices]
        ndcg_scores.append(compute_ndcg(relevance, k=5))

        total_queries += 1

    r_at_1 = recall_at_1 / total_queries if total_queries > 0 else 0.0
    r_at_5 = recall_at_5 / total_queries if total_queries > 0 else 0.0
    ndcg_at_5 = np.mean(ndcg_scores) if ndcg_scores else 0.0

    results_df = pd.DataFrame([{
        "model_path": model_path,
        "benchmark_csv": benchmark_csv,
        "R@1": r_at_1,
        "R@5": r_at_5,
        "NDCG@5": ndcg_at_5,
        "total_queries": total_queries,
        "total_dishes": len(dish_names),
    }])

    if os.path.exists(output_csv):
        results_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")

    return results_df.iloc[0].to_dict()


if __name__ == "__main__":
    results = evaluate_model(MODEL_PATH, BENCHMARK_CSV, DEVICE, OUTPUT_CSV)
