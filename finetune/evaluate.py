import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from vllm import LLM

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "google/siglip2-base-patch16-224")
BENCHMARK_CSV = os.getenv("BENCHMARK_CSV", "test.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "evaluation_results.csv")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.5"))


def get_text_embedding(llm: LLM, text: str) -> np.ndarray:
    """Get normalized text embedding using vLLM."""
    out = llm.embed([text])
    embed = np.asarray(out[0].outputs.embedding, dtype="float32")
    embed = embed / np.linalg.norm(embed)
    return embed[None, :]  # Shape: (1, dim)


def get_image_embedding(llm: LLM, image_path: str) -> np.ndarray:
    """Get normalized image embedding using vLLM."""
    image = Image.open(image_path).convert("RGB")
    out = llm.embed([image])
    embed = np.asarray(out[0].outputs.embedding, dtype="float32")
    embed = embed / np.linalg.norm(embed)
    return embed[None, :]  # Shape: (1, dim)


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


def evaluate_model(model_path, benchmark_csv, output_csv, gpu_memory_utilization):
    """Evaluate model on benchmark dataset using vLLM."""
    print(f"Model: {model_path}")
    print(f"Dataset: {benchmark_csv}")
    print(f"GPU Memory: {gpu_memory_utilization}")
    print("-" * 60)

    # Load model with vLLM
    llm = LLM(
        model=model_path,
        task="embed",
        tokenizer_mode="auto",
        gpu_memory_utilization=gpu_memory_utilization,
    )

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
            dish_text_embeddings[dish_name] = get_text_embedding(llm, dish_text)

        if dish_image:
            dish_image_embeddings[dish_name] = get_image_embedding(llm, dish_image)

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
            query_text_embed = get_text_embedding(llm, query_text)
            text_similarities = cosine_similarity(query_text_embed, dish_text_matrix)[0]
            for dish_name, score in zip(dish_names, text_similarities):
                combined_scores[dish_name] = combined_scores.get(
                    dish_name, 0.0
                ) + float(score)

        if query_image and dish_image_matrix is not None:
            query_image_embed = get_image_embedding(llm, query_image)
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
        MODEL_PATH, BENCHMARK_CSV, OUTPUT_CSV, GPU_MEMORY_UTILIZATION
    )
