#!/usr/bin/env python3
"""
NanoBEIR evaluator with one-query debug peek.

- Optional baseline via Sentence-Transformers
- llama.cpp --embedding HTTP server (/embedding)
- Prints a sanity check for one query so you can see relevant IDs vs retrieved IDs
"""

import os, json, math, time, statistics, argparse, random
from typing import List, Dict, Any, Iterable, Union

# ----------------- Tunables -----------------
MAX_CHARS = 2000
RETRY_BACKOFF = [0.0, 0.2, 0.6]
RETRY_SHRINK  = [1.00, 0.60, 0.35]

# ----------------- Small utilities -----------------
def p95(latencies_ms: List[float]) -> float:
    if not latencies_ms:
        return 0.0
    return float(statistics.quantiles(latencies_ms, n=100)[94])

def time_calls(fn, payloads: List[Any], warmup: int = 3) -> float:
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
    return float(ranks[m//2] if m % 2 else (ranks[m//2 - 1] + ranks[m//2]) / 2)

# ----------------- Minimal BEIR imports -----------------
def _imports():
    global util, GenericDataLoader, EvaluateRetrieval, DRES
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

# ----------------- Dataset URLs -----------------
_BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "fiqa":    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
}

def load_dataset(name: str, base_dir: str = "./datasets"):
    _imports()
    url_or_id = _BEIR_URLS.get(name, name)
    data_path = util.download_and_unzip(url_or_id, base_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

# ----------------- Model wrappers -----------------
def _doc_text(doc: Dict[str, str]) -> str:
    title = (doc.get("title") or "").strip()
    text  = (doc.get("text")  or "").strip()
    merged = (title + " " + text).strip()
    return merged[:MAX_CHARS] if len(merged) > MAX_CHARS else merged

class STModel:
    def __init__(self, hf_id: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(hf_id)

    def encode_queries(self, queries, batch_size=32, **kwargs):
        return self.model.encode(
            queries, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False
        )

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        docs_iter = corpus.values() if isinstance(corpus, dict) else corpus
        texts = [((d.get("title") or "") + " " + (d.get("text") or "")).strip()[:MAX_CHARS]
                 for d in docs_iter]
        return self.model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True
        )

class LlamaServerEncoder:
    """Client for llama.cpp --embedding server (/embedding). Ensures fixed-dim vectors."""
    def __init__(self, endpoint: str):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.2, status_forcelist=(502, 503, 504))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.endpoint = endpoint.rstrip("/")
        self._dim = None  # lock the embedding size after first good response

    def _parse_embedding_json(self, js):
        # try to dig out the first numeric list from many shapes
        def first_vector(obj):
            if isinstance(obj, dict):
                # common fields
                for k in ("embedding", "vector", "values", "data"):
                    if k in obj:
                        return first_vector(obj[k])
                # otherwise try the first value
                if obj:
                    return first_vector(next(iter(obj.values())))
                return []
            if isinstance(obj, (list, tuple)):
                if not obj:
                    return []
                if isinstance(obj[0], dict):
                    return first_vector(obj[0])
                if isinstance(obj[0], (list, tuple)):
                    return first_vector(obj[0])
                return obj
            return [obj]
        return first_vector(js)

    def _to_1d_numeric(self, obj):
        def flatten(xs):
            for x in xs:
                if isinstance(x, (list, tuple)):
                    yield from flatten(x)
                elif isinstance(x, dict):
                    for k in ("vector", "embedding", "values", "data"):
                        if k in x:
                            yield from flatten(x[k]); break
                    else:
                        for v in x.values():
                            yield from flatten(v)
                else:
                    yield x
        vec = list(flatten(self._parse_embedding_json(obj)))
        vec = [float(x) for x in vec]
        return vec

    def _normalize_vec(self, vec):
        vec = self._to_1d_numeric(vec)
        if self._dim is None:
            self._dim = len(vec)
            print(f"[llama] locked embedding dim = {self._dim}", flush=True)
        if len(vec) < self._dim:
            vec = vec + [0.0] * (self._dim - len(vec))
        elif len(vec) > self._dim:
            vec = vec[:self._dim]
        return vec

    def _embed_one(self, text: str, timeout=60):
        orig = text
        for backoff, shrink in zip(RETRY_BACKOFF, RETRY_SHRINK):
            if backoff: time.sleep(backoff)
            t = orig[: int(MAX_CHARS * shrink)]
            r = self.session.post(self.endpoint, json={"content": t}, timeout=timeout)
            if r.status_code >= 500:
                continue
            r.raise_for_status()
            return self._normalize_vec(r.json())
        # final tiny attempt
        t = orig[: min(512, len(orig))]
        r = self.session.post(self.endpoint, json={"content": t}, timeout=timeout)
        r.raise_for_status()
        return self._normalize_vec(r.json())

    def encode_queries(self, queries, batch_size=1, **kwargs):
        return [self._embed_one(q) for q in queries]

    def encode_corpus(self, corpus, batch_size=1, **kwargs):
        docs_iter = corpus.values() if isinstance(corpus, dict) else corpus
        texts = [_doc_text(d) for d in docs_iter]
        return [self._embed_one(t) for t in texts]

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
    print(f"Relevant doc IDs ({len(relevant)}): {sorted(list(relevant))[:10]}{' ...' if len(relevant)>10 else ''}")
    print("\nTop-{} retrieved:".format(k))
    for i, (did, score) in enumerate(ranked, 1):
        title = (corpus.get(did, {}).get('title') if isinstance(corpus, dict) else None) or ""
        print(f"{i:>2}. {did}  score={score:.4f}  title={title[:80]}")
    print(f"\nOverlap@{k}: {sorted(list(overlap)) if overlap else 'None'}")
    print("=====================================\n", flush=True)

def evaluate_model(beir_model, corpus, queries, qrels, cost_per_1k: float):
    _imports()
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
            ranked = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:k]
            retrieved_ids = {did for did, _ in ranked}
            if relevant & retrieved_ids:
                hits += 1
        return (hits / total) if total > 0 else 0.0

    R1  = recall_at_k(1)
    R5  = recall_at_k(5)
    R10 = recall_at_k(10)

    print(f"[debug] R@1={R1:.3f}  R@5={R5:.3f}  R@10={R10:.3f}", flush=True)

    # --- NDCG@10 from BEIR (robust pick of the key) ---
    ndcg, _map, recall_dict, precision = retriever.evaluate(qrels, results, retriever.k_values)
    def _pick_ndcg10(d):
        for key in (10, "10", "ndcg@10", "NDCG@10"):
            if key in d:
                return float(d[key])
        # sometimes BEIR nests like {'NDCG': {10: val, ...}}
        if isinstance(d, dict):
            for v in d.values():
                try:
                    return _pick_ndcg10(v)
                except Exception:
                    pass
        return 0.0
    N10 = _pick_ndcg10(ndcg)

    # --- Median rank of first relevant (as before) ---
    medr = float(median_rank(results, qrels))

    # --- Latency p95 on first ~50 queries ---
    qids = list(queries.keys())
    probe = [queries[q] for q in qids[:min(50, len(qids))]]
    lat = time_calls(lambda batch: beir_model.encode_queries(batch, batch_size=1), probe, warmup=3)

    return {
        "R@1": float(R1),
        "R@5": float(R5),
        "R@10": float(R10),
        "NDCG@10": float(N10),
        "MedR": float(medr),
        "Latency_ms_p95": float(lat),
        "Cost_per_1k_queries": float(cost_per_1k),
    }


# ----------------- Main -----------------
def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("output_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)
    leaderboard_path = cfg.get("leaderboard_json", f"{out_dir}/nanobeir_leaderboard.json")

    datasets = cfg.get("datasets", ["scifact"])
    cost = cfg.get("cost_per_1k", {})
    models_cfg = cfg.get("models", {})

    all_out: Dict[str, Dict[str, Any]] = {}

    for ds in datasets:
        corpus, queries, qrels = load_dataset(ds)
        all_out.setdefault(ds, {})

        if "baseline" in models_cfg:
            m_id = models_cfg["baseline"]["hf_id"]
            print(f"[baseline] {m_id}")
            out = evaluate_model(STModel(m_id), corpus, queries, qrels, cost.get("baseline", 0.0))
            all_out[ds]["baseline"] = out

        if "qwen_edge_llamaserver" in models_cfg:
            ep = models_cfg["qwen_edge_llamaserver"]["endpoint"]
            print(f"[qwen_edge_llamaserver] {ep}")
            out = evaluate_model(LlamaServerEncoder(ep), corpus, queries, qrels,
                                 cost.get("qwen_edge_llamaserver", 0.0))
            all_out[ds]["qwen_edge_llamaserver"] = out

    with open(leaderboard_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"[saved] {leaderboard_path}")
    print(json.dumps(all_out, indent=2))

if __name__ == "__main__":
    main()
