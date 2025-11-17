import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoProcessor


class EmbeddingBenchmark:
    def __init__(self, model_path, benchmark_csv):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.df = pd.read_csv(benchmark_csv)

    def get_embeddings(self, text: str | None = None, image_path: str | None = None):
        with torch.no_grad():
            if text is not None and image_path is not None:
                assert image_path is not None
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(
                    text=text, images=image, return_tensors="pt", padding=True
                )
            elif text is not None:
                inputs = self.processor(text=text, return_tensors="pt", padding=True)
            elif image_path is not None:
                assert image_path is not None
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
            else:
                raise ValueError("Either text or image_path must be provided")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            if text is not None and image_path is not None:
                text_embed = outputs.text_embeds.cpu().numpy()
                image_embed = outputs.image_embeds.cpu().numpy()
                return text_embed, image_embed
            elif text is not None:
                return outputs.text_embeds.cpu().numpy()
            else:
                return outputs.image_embeds.cpu().numpy()

    def compute_retrieval_metrics(self, k_values=[1, 3, 5]):
        results = {}

        dish_embeddings = {}
        for dish_name in self.df["dish_name"].unique():
            dish_rows = self.df[self.df["dish_name"] == dish_name]
            dish_image = dish_rows.iloc[0]["image"]
            dish_embeddings[dish_name] = self.get_embeddings(image_path=dish_image)

        dish_names = list(dish_embeddings.keys())
        dish_embeds_matrix = np.vstack([dish_embeddings[name] for name in dish_names])

        correct_at_k = {k: 0 for k in k_values}
        total_queries = 0

        for idx, row in self.df.iterrows():
            query_text = None
            if bool(pd.notna(row["query"])):
                text_val = str(row["query"]).strip()
                if text_val:
                    query_text = text_val

            query_image = str(row["image"]) if bool(pd.notna(row["image"])) else None
            true_dish = str(row["dish_name"])

            if query_text is not None and query_image is not None:
                text_embed, image_embed = self.get_embeddings(
                    text=query_text, image_path=query_image
                )
                query_embed = (text_embed + image_embed) / 2
            elif query_text is not None:
                query_embed = self.get_embeddings(text=query_text)
            elif query_image is not None:
                query_embed = self.get_embeddings(image_path=query_image)
            else:
                continue

            similarities = cosine_similarity(query_embed, dish_embeds_matrix)[0]
            top_k_indices = np.argsort(similarities)[::-1]

            for k in k_values:
                top_k_dishes = [dish_names[i] for i in top_k_indices[:k]]
                if true_dish in top_k_dishes:
                    correct_at_k[k] += 1

            total_queries += 1

        for k in k_values:
            results[f"recall@{k}"] = (
                correct_at_k[k] / total_queries if total_queries > 0 else 0
            )

        results["total_queries"] = total_queries
        results["total_dishes"] = len(dish_names)

        return results

    def compute_mrr(self):
        dish_embeddings = {}
        for dish_name in self.df["dish_name"].unique():
            dish_rows = self.df[self.df["dish_name"] == dish_name]
            dish_image = dish_rows.iloc[0]["image"]
            dish_embeddings[dish_name] = self.get_embeddings(image_path=dish_image)

        dish_names = list(dish_embeddings.keys())
        dish_embeds_matrix = np.vstack([dish_embeddings[name] for name in dish_names])

        reciprocal_ranks = []

        for idx, row in self.df.iterrows():
            query_text = None
            if bool(pd.notna(row["query"])):
                text_val = str(row["query"]).strip()
                if text_val:
                    query_text = text_val

            query_image = str(row["image"]) if bool(pd.notna(row["image"])) else None
            true_dish = str(row["dish_name"])

            if query_text is not None and query_image is not None:
                text_embed, image_embed = self.get_embeddings(
                    text=query_text, image_path=query_image
                )
                query_embed = (text_embed + image_embed) / 2
            elif query_text is not None:
                query_embed = self.get_embeddings(text=query_text)
            elif query_image is not None:
                query_embed = self.get_embeddings(image_path=query_image)
            else:
                continue

            similarities = cosine_similarity(query_embed, dish_embeds_matrix)[0]
            ranked_dishes = [dish_names[i] for i in np.argsort(similarities)[::-1]]

            if true_dish in ranked_dishes:
                rank = ranked_dishes.index(true_dish) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def run_benchmark(model_path, benchmark_csv):
    print(f"Benchmarking model: {model_path}")
    print(f"Using dataset: {benchmark_csv}")
    print("-" * 60)

    benchmark = EmbeddingBenchmark(model_path, benchmark_csv)

    retrieval_metrics = benchmark.compute_retrieval_metrics(k_values=[1, 3, 5, 10])
    mrr = benchmark.compute_mrr()

    print("\nRetrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        if "recall" in metric:
            print(f"  {metric}: {value:.4f} ({value * 100:.2f}%)")
        else:
            print(f"  {metric}: {value}")

    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")
    print("-" * 60)

    return {**retrieval_metrics, "mrr": mrr}


if __name__ == "__main__":
    baseline_results = run_benchmark("google/siglip-base-patch16-224", "test.csv")

    print("\n" + "=" * 60)
    print("To benchmark your finetuned model, run:")
    print('run_benchmark("finetuned_siglip", "test.csv")')
    print("=" * 60)
