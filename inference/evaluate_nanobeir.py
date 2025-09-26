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

# --- Third-Party Imports ---
import torch
import yaml
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
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
    url_or_id = _BEIR_URLS.get(name, name)
    data_path = util.download_and_unzip(url_or_id, base_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
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

    return {
        "R@1": float(r1),
        "R@5": float(r5),
        "R@10": float(r10),
        "NDCG@10": float(n10),
        "MedR": float(medr),
        "Latency_ms_p95": float(lat),
        "Cost_per_1k_queries": float(cost_per_1k),
    }


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
