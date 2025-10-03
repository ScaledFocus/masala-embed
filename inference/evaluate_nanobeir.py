#!/usr/bin/env python3
"""
NanoBEIR evaluator with one-query debug peek.

- Optional baseline via Sentence-Transformers
- llama.cpp --embedding HTTP server (/embedding)
- Prints a sanity check for one query so you can see relevant IDs vs retrieved IDs
"""

# --- Python Standard Library Imports ---
import argparse
import json
import os
import random
import statistics
import time
from typing import Any

import pandas as pd

# --- Third-Party Imports ---
import torch
import yaml
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from huggingface_hub import snapshot_download
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# ----------------- Tunables -----------------
MAX_CHARS = 2000
RETRY_BACKOFF = [0.0, 0.2, 0.6]
RETRY_SHRINK = [1.00, 0.60, 0.35]


# ----------------- Small utilities -----------------
def p95(latencies_ms: list[float]) -> float:
    if not latencies_ms:
        return 0.0
    return float(statistics.quantiles(latencies_ms, n=100)[94])


def measure_throughput(model, queries: list[str], batch_size: int = 32) -> float:
    """Measures true batch throughput in Queries Per Second."""
    if not queries:
        return 0.0

    # Warmup run
    _ = model.encode_queries(queries[:batch_size], batch_size=batch_size)

    print(f"[debug] Measuring throughput on {len(queries)} queries...")
    t_start = time.time()
    _ = model.encode_queries(queries, batch_size=batch_size)
    t_end = time.time()

    total_time = t_end - t_start
    if total_time == 0:
        return 0.0

    qps = len(queries) / total_time
    print(f"[debug] Throughput: {qps:.2f} QPS")
    return qps


def time_calls(fn, payloads: list[Any], warmup: int = 3) -> float:
    if not payloads:
        return 0.0
    for _ in range(max(0, warmup)):
        _ = fn([payloads[0]])
    lat = []
    for p in payloads:
        t0 = time.time()
        _ = fn([p])
        t1 = time.time()
        lat.append((t1 - t0) * 1000.0)
    return p95(lat)


def median_rank(results, qrels) -> float:
    ranks = []
    for qid, doc_scores in results.items():
        if qid not in qrels:
            continue
        relevant = {did for did, rel in qrels[qid].items() if rel > 0}
        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
        for idx, (did, _) in enumerate(ranked, start=1):
            if did in relevant:
                ranks.append(idx)
                break
    if not ranks:
        return 0.0
    ranks.sort()
    m = len(ranks)
    return float(ranks[m // 2] if m % 2 else (ranks[m // 2 - 1] + ranks[m // 2]) / 2)


# ----------------- Dataset URLs -----------------
_BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
}


def load_dataset(name: str, base_dir: str = "./datasets"):
    print(f"[dataset] Attempting to load dataset: '{name}'")

    if "/" not in name:
        raise ValueError(
            f"'{name}' must be a Hugging Face dataset ID like 'user/repo'."
        )

    # --- Step 1: Download the raw files from the Hub ---
    print(f"-> Downloading '{name}' from the Hub...")
    try:
        local_path = snapshot_download(repo_id=name, repo_type="dataset")
        print(f"--> Dataset downloaded to cache: {local_path}")
    except Exception as e:
        error_message = (
            f"\n[ERROR] Failed to download repository '{name}'. "
            "Please double-check the name in your YAML config."
        )
        print(error_message)
        raise e

    # --- Step 2: Convert and Sanitize files from Parquet to BEIR format ---
    print("--> Downloaded Parquet files. Converting & sanitizing to BEIR format...")

    # Convert Corpus
    corpus_parquet_path = os.path.join(
        local_path, "corpus", "train-00000-of-00001.parquet"
    )
    corpus_jsonl_path = os.path.join(local_path, "corpus.jsonl")
    if not os.path.exists(corpus_jsonl_path):
        corpus_df = pd.read_parquet(corpus_parquet_path)

        # --- THE BULLETPROOF FIX ---
        print("--> Applying final sanitization to corpus...")
        # 1. Ensure a 'title' column exists. If not, create it with empty strings.
        if "title" not in corpus_df.columns:
            print("--> 'title' column not found. Creating empty 'title' column.")
            corpus_df["title"] = ""

        # 2. Sanitize both 'title' and 'text' columns to replace any None values.
        corpus_df["title"] = corpus_df["title"].fillna("")
        corpus_df["text"] = corpus_df["text"].fillna("")

        corpus_df.to_json(corpus_jsonl_path, orient="records", lines=True)
        print(f"--> Converted and saved {corpus_jsonl_path}")

    # Convert Queries (Sanitization for safety)
    queries_parquet_path = os.path.join(
        local_path, "queries", "train-00000-of-00001.parquet"
    )
    queries_jsonl_path = os.path.join(local_path, "queries.jsonl")
    if not os.path.exists(queries_jsonl_path):
        queries_df = pd.read_parquet(queries_parquet_path)
        queries_df["text"] = queries_df["text"].fillna("")
        queries_df.to_json(queries_jsonl_path, orient="records", lines=True)
        print(f"--> Converted and saved {queries_jsonl_path}")

    # Convert Qrels
    qrels_parquet_path = os.path.join(
        local_path, "qrels", "train-00000-of-00001.parquet"
    )
    qrels_dir = os.path.join(local_path, "qrels")
    qrels_tsv_path = os.path.join(qrels_dir, "test.tsv")
    if not os.path.exists(qrels_tsv_path):
        os.makedirs(qrels_dir, exist_ok=True)
        qrels_df = pd.read_parquet(qrels_parquet_path)
        if "score" not in qrels_df.columns:
            qrels_df["score"] = 1
        qrels_df = qrels_df.rename(
            columns={
                "query_id": "query-id",
                "corpus_id": "corpus-id",
            }
        )
        qrels_df.to_csv(qrels_tsv_path, sep="\t", index=False)
        print(f"--> Converted and saved {qrels_tsv_path}")

    # --- Step 3: Load the newly converted, BEIR-compatible files ---
    print("--> Conversion complete. Loading formatted data...")
    corpus, queries, qrels = GenericDataLoader(data_folder=local_path).load(
        split="test"
    )
    return corpus, queries, qrels


# ----------------- Model wrappers -----------------
def _doc_text(doc: dict[str, str]) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    merged = (title + " " + text).strip()
    return merged[:MAX_CHARS] if len(merged) > MAX_CHARS else merged


class STModel:
    def __init__(self, hf_id: str):
        self.model = SentenceTransformer(hf_id)

    def encode_queries(self, queries, batch_size=32, **kwargs):
        return self.model.encode(
            queries,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        docs_iter = corpus.values() if isinstance(corpus, dict) else corpus
        texts = [
            ((d.get("title") or "") + " " + (d.get("text") or "")).strip()[:MAX_CHARS]
            for d in docs_iter
        ]
        return self.model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True
        )


# Add this entire class to your evaluate_nanobeir.py script


class QwenVLModel:
    """
    Custom model loader for a Qwen Vision-Language model to extract text embeddings.
    This uses the main transformers library instead of sentence-transformers.
    """

    def __init__(self, hf_id: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            hf_id, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        print(f"Loaded {hf_id} on {self.device} with dtype {self.model.dtype}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # This model doesn't use a simple tokenizer, it uses a processor
        # that handles both text and images. We are only providing text.
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            # Get the model's hidden states
            outputs = self.model(**inputs, output_hidden_states=True)

            # Use the last hidden state
            last_hidden_state = outputs.hidden_states[-1]

            # Perform mean pooling to get a single vector per text
            # (This is a standard way to get sentence embeddings)
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(last_hidden_state.size())
                .float()
            )
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled_embeddings = sum_embeddings / sum_mask

        # Normalize the embeddings
        normalized_embeddings = torch.nn.functional.normalize(
            pooled_embeddings, p=2, dim=1
        )

        return normalized_embeddings.cpu().tolist()

    def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs):
        return self._embed(queries)

    def encode_corpus(self, corpus: list[dict], batch_size: int = 32, **kwargs):
        # The BEIR library passes the corpus as a list of dictionaries
        texts = [doc.get("title", "") + " " + doc.get("text", "") for doc in corpus]
        return self._embed(texts)


class QwenHFEncoder:
    """
    HF encoder for Qwen/qwen3-embedding-0.6b (or compatible).
    - Handles dict or list corpora (BEIR can pass either)
    - Mean-pools token embeddings
    - Optional L2 normalization
    """

    def __init__(
        self,
        hf_id: str,
        pooling: str = "mean",
        normalize: bool = True,
        max_len: int = 512,
    ):
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.normalize = normalize
        self.max_len = max_len

        # Use slow tokenizer + trust_remote_code to avoid the fast-tokenizer JSON error
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, use_fast=False, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(
            self.device
        )
        self.model.eval()

    def _encode_batch(self, texts, batch_size=32):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)

            with self.torch.no_grad():
                out = self.model(**toks)

            # Prefer last_hidden_state, fall back to pooler_output if present
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                x = out.last_hidden_state  # [B, T, H]
                # attention mask-based mean pooling
                mask = toks.attention_mask.unsqueeze(-1).type_as(x)  # [B, T, 1]
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, H]
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                x = out.pooler_output  # [B, H]
            else:
                # very defensive fallback
                x = out[0] if isinstance(out, (list | tuple)) else out

            if self.normalize:
                x = self.torch.nn.functional.normalize(x, p=2, dim=-1)

            embs.extend(x.detach().cpu().tolist())
        return embs

    def encode_queries(self, queries, batch_size=32, **kwargs):
        # queries is a list[str]
        return self._encode_batch(list(queries), batch_size=batch_size)

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        """
        corpus can be:
          - dict: {doc_id: {"title":..., "text":...}, ...}
          - list: [{"title":..., "text":...}, ...]
        Return must be a list of embeddings aligned to the order BEIR provides.
        """
        if isinstance(corpus, dict):
            docs_iter = corpus.values()
        elif isinstance(corpus, list):
            docs_iter = corpus
        else:
            raise TypeError(f"Unsupported corpus type: {type(corpus)}")

        def _merge(doc):
            title = (doc.get("title") or "").strip()
            text = (doc.get("text") or "").strip()
            s = (title + " " + text).strip()
            return s[:2000] if len(s) > 2000 else s

        texts = [_merge(d) for d in docs_iter]
        return self._encode_batch(texts, batch_size=batch_size)


# ----------------- Evaluation + Debug Peek -----------------
def debug_one_query(corpus, queries, qrels, results, k=5):
    """Print one query's relevant IDs vs. top-k retrieved IDs."""
    # pick a query that has ground-truth and results
    candidate_qids = [qid for qid in qrels.keys() if qid in results]
    if not candidate_qids:
        print("[debug] No overlapping queries between qrels and results.")
        return
    qid = random.choice(candidate_qids)
    q_text = queries.get(qid, "(query text missing)")
    relevant = {did for did, rel in qrels[qid].items() if rel > 0}

    ranked = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:k]
    retrieved_ids = [did for did, _ in ranked]
    overlap = relevant.intersection(retrieved_ids)

    print("\n============= DEBUG PEEK =============")
    print(f"Query ID: {qid}")
    print(f"Query: {q_text}")
    print(
        f"Relevant doc IDs ({len(relevant)}): {sorted(list(relevant))[:10]}"
        f"{' ...' if len(relevant) > 10 else ''}"
    )
    print(f"\nTop-{k} retrieved:")
    for i, (did, score) in enumerate(ranked, 1):
        title = (
            corpus.get(did, {}).get("title") if isinstance(corpus, dict) else None
        ) or ""
        print(f"{i:>2}. {did}  score={score:.4f}  title={title[:80]}")
    print(f"\nOverlap@{k}: {sorted(list(overlap)) if overlap else 'None'}")
    print("=====================================\n", flush=True)


def evaluate_model(beir_model, corpus, queries, qrels, cost_per_1k: float):
    retriever_model = DRES(beir_model, batch_size=32)
    retriever = EvaluateRetrieval(retriever_model, score_function="cos_sim")

    # Run retrieval
    results = retriever.retrieve(corpus, queries)

    # --- Manual Recall@K (hit rate) ---
    def recall_at_k(k: int) -> float:
        hits = 0
        total = 0
        for qid, rels in qrels.items():
            if qid not in results:
                continue
            total += 1
            relevant = {did for did, rel in rels.items() if rel > 0}
            ranked = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[
                :k
            ]
            retrieved_ids = {did for did, _ in ranked}
            if relevant & retrieved_ids:
                hits += 1
        return (hits / total) if total > 0 else 0.0

    r1 = recall_at_k(1)
    r5 = recall_at_k(5)
    r10 = recall_at_k(10)

    print(f"[debug] R@1={r1:.3f}  R@5={r5:.3f}  R@10={r10:.3f}", flush=True)

    # --- NDCG@10 from BEIR (robust pick of the key) ---
    ndcg, _, _, _ = retriever.evaluate(qrels, results, retriever.k_values)

    def _pick_ndcg10(d):
        for key in (10, "10", "ndcg@10", "NDCG@10"):
            if key in d:
                return float(d[key])
        # sometimes BEIR nests like {'NDCG': {10: val, ...}}
        if isinstance(d, dict):
            for v in d.values():
                try:
                    return _pick_ndcg10(v)
                except (TypeError, KeyError):
                    pass
        return 0.0

    n10 = _pick_ndcg10(ndcg)

    # --- Median rank of first relevant (as before) ---
    medr = float(median_rank(results, qrels))

    # --- Latency p95 on first ~50 queries ---
    qids = list(queries.keys())
    probe = [queries[q] for q in qids[: min(50, len(qids))]]
    lat = time_calls(
        lambda batch: beir_model.encode_queries(batch, batch_size=1), probe, warmup=3
    )
    # lat_p95 = float(lat)
    # est_qps = 1000.0 / lat_p95 if lat_p95 > 0 else 0.0
    all_query_texts = list(queries.values())
    true_qps = measure_throughput(beir_model, all_query_texts)
    return {
        "R@1": float(r1),
        "R@5": float(r5),
        "R@10": float(r10),
        "NDCG@10": float(n10),
        "MedR": float(medr),
        "Latency_ms_p95": float(lat),
        # "Est_Throughput_QPS": float(est_qps),
        "True_Throughput_QPS": float(true_qps),
    }


# ... (inside evaluate_nanobier.py)


class VLLMAPIEncoder:
    """
    Client for a vLLM OpenAI-compatible server that provides embeddings.
    - Handles retry logic for robustness.
    """

    def __init__(self, base_url: str, model_name: str, max_len: int = 512):
        # NOTE: vLLM embed API uses the client, but is simpler than chat/completion.
        self.client = OpenAI(base_url=base_url, api_key="sk-not-used-by-vllm-server")
        self.model_name = model_name
        self.max_len = max_len
        self.normalize = True

    def _merge_and_truncate(self, doc_list):
        # Helper function to preprocess text the same way
        texts = []
        for d in doc_list:
            title = (d.get("title") or "").strip()
            text = (d.get("text") or "").strip()
            s = (title + " " + text).strip()
            texts.append(s[: self.max_len] if len(s) > self.max_len else s)
        return texts

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        # Using the OpenAI-compatible embedding endpoint
        # Add retry logic to handle transient network/server issues
        for delay in RETRY_BACKOFF:
            time.sleep(delay)
            try:
                # The input list of strings is sent to the /v1/embeddings endpoint
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )

                # The response is typically a list of dicts with 'embedding' key
                # We extract the embeddings and return them as a list of lists (vectors)
                embeddings = [d.embedding for d in response.data]
                return embeddings

            except Exception as e:
                print(f"[vLLM-API-ERROR] Failed to get embeddings: {e}. Retrying...")
                continue

        # If all retries fail, raise the last exception or return an error state
        raise RuntimeError("vLLM API failed after all retries.")

    def encode_queries(self, queries, batch_size=32, **kwargs):
        # queries is a list[str]. Note: vLLM handles the batching internally
        return self._call_api(list(queries))

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        # corpus is dict or list (BEIR-compatible format)
        if isinstance(corpus, dict):
            docs_iter = corpus.values()
        elif isinstance(corpus, list):
            docs_iter = corpus
        else:
            raise TypeError(f"Unsupported corpus type: {type(corpus)}")

        texts = self._merge_and_truncate(docs_iter)
        # Note: vLLM API server is better at handling large corpus input than
        # the local HF encoder, but for robustness with huge datasets, you
        # might need to add chunking/batching around _call_api if the number
        # of documents exceeds vLLM's max batch size (which is usually very large).
        return self._call_api(texts)


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("output_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)
    leaderboard_path = cfg.get(
        "leaderboard_json", f"{out_dir}/nanobeir_leaderboard.json"
    )

    datasets = cfg.get("datasets", ["scifact"])
    cost = cfg.get("cost_per_1k", {})
    models_cfg = cfg.get("models", {})

    all_out: dict[str, dict[str, Any]] = {}

    for ds in datasets:
        corpus, queries, qrels = load_dataset(ds)
        all_out.setdefault(ds, {})

        if "baseline" in models_cfg:
            m_id = models_cfg["baseline"]["hf_id"]
            print(f"[baseline] {m_id}")
            out = evaluate_model(
                STModel(m_id), corpus, queries, qrels, cost.get("baseline", 0.0)
            )
            all_out[ds]["baseline"] = out

        if "colqwen_baseline" in models_cfg:
            m_id = models_cfg["colqwen_baseline"]["hf_id"]
            print(f"[colqwen_baseline] {m_id}")
            out = evaluate_model(
                QwenVLModel(m_id),
                corpus,
                queries,
                qrels,
                cost.get("colqwen_baseline", 0.0),
            )
            all_out[ds]["colqwen_baseline"] = out

        if "vllm_api_encoder" in models_cfg:
            m = models_cfg["vllm_api_encoder"]
            base_url = m.get("base_url", "http://localhost:8000/v1")
            model_name = m.get("model_name", "Qwen/qwen3-embedding-0.6b")
            max_len = m.get("max_len", 512)
            print(
                f"[vllm_api_encoder] {model_name} | base_url={base_url} "
                f"max_len={max_len}"
            )
            encoder = VLLMAPIEncoder(
                base_url=base_url, model_name=model_name, max_len=max_len
            )
            out = evaluate_model(
                encoder, corpus, queries, qrels, cost.get("vllm_api_encoder", 0.0)
            )
            all_out[ds]["vllm_api_encoder"] = out

        if "qwen_embedding" in models_cfg:
            m = models_cfg["qwen_embedding"]
            hf_id = m.get("hf_id", "Qwen/qwen3-embedding-0.6b")
            pooling = m.get("pooling", "mean")
            normalize = m.get("normalize", True)
            max_len = m.get("max_len", 512)
            print(
                f"[qwen_embedding] {hf_id} | pooling={pooling} "
                f"normalize={normalize} max_len={max_len}"
            )
            encoder = QwenHFEncoder(
                hf_id, pooling=pooling, normalize=normalize, max_len=max_len
            )
            out = evaluate_model(
                encoder, corpus, queries, qrels, cost.get("qwen_embedding", 0.0)
            )
            all_out[ds]["qwen_embedding"] = out

    with open(leaderboard_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"[saved] {leaderboard_path}")
    print(json.dumps(all_out, indent=2))


if __name__ == "__main__":
    main()
